import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from world import World
from protocol import Protocol
from events import Event, DiscardQubitEvent, EntanglementSwappingEvent, SourceEvent
from copy import copy
from libs.aux_functions import apply_single_qubit_map, y_noise_channel, z_noise_channel, w_noise_channel, distance
from consts import SPEED_OF_LIGHT_IN_VACCUM as C
from consts import AVERAGE_EARTH_RADIUS as R_E
from consts import ETA_ATM_PI_HALF_780_NM
import libs.matrix as mat
from functools import lru_cache
from quantum_objects import SchedulingSource, Station
from noise import NoiseModel, NoiseChannel


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return dephasing_noise_channel


def construct_y_noise_channel(epsilon):
    return lambda rho: y_noise_channel(rho=rho, epsilon=epsilon)


def construct_w_noise_channel(epsilon):
    return lambda rho: w_noise_channel(rho=rho, alpha=(1 - epsilon))


def alpha_of_eta(eta, p_d):
    return eta * (1 - p_d) / (1 - (1 - eta) * (1 - p_d)**2)


def eta_dif(distance, divergence_half_angle, sender_aperture_radius, receiver_aperture_radius):
    # calculated by simple geometry, because gaussian effects do not matter much
    x = sender_aperture_radius + distance * np.tan(divergence_half_angle)
    arriving_fraction = receiver_aperture_radius**2 / x**2
    if arriving_fraction > 1:
        arriving_fraction = 1
    return arriving_fraction


def eta_atm(elevation):
    # eta of pi/2 to the power of csc(theta), equation (A4) in https://arxiv.org/abs/2006.10636
    # eta of pi/2 (i.e. straight up) is ~0.8 for 780nm wavelength.
    if elevation < 0:
        return 0
    return ETA_ATM_PI_HALF_780_NM**(1 / np.sin(elevation))


def sat_dist_curved(ground_dist, h):
    # ground dist refers to distance between station and the "shadow" of the satellite
    alpha = ground_dist / R_E
    L = np.sqrt(R_E**2 + (R_E + h)**2 - 2 * R_E * (R_E + h) * np.cos(alpha))
    return L


def elevation_curved(ground_dist, h):
    # ground dist refers to distance between station and the "shadow" of the satellite
    alpha = ground_dist / R_E
    L = np.sqrt(R_E**2 + (R_E + h)**2 - 2 * R_E * (R_E + h) * np.cos(alpha))
    beta = np.arcsin(R_E / L * np.sin(alpha))
    gamma = np.pi - alpha - beta
    return gamma - np.pi / 2


class RepeatingEvent(Event):  # still an abstract class
    def __init__(self, time, offset_time, required_objects=[], priority=20, ignore_blocked=False):
        self.offset_time = offset_time
        super(RepeatingEvent, self).__init__(time=time, required_objects=required_objects, priority=priority, ignore_blocked=ignore_blocked)

    def resolve(self):
        return_value = super(RepeatingEvent, self).resolve()
        if self.event_queue is None:
            raise ValueError("RepeatingEvent is not associated with an event_queue and cannot be rescheduled.")
        new_event = copy(self)
        new_event.time = self.event_queue.current_time + self.offset_time
        # register new event with objects
        for required_object in new_event.required_objects:
            assert required_object in required_object.world
            required_object.required_by_events += [new_event]
        self.event_queue.add_event(new_event)
        return return_value


class SendOnScheduleEvent(RepeatingEvent):
    def __init__(self, time, offset_time, station, target_station, channel_eta_func, priority=21, ignore_blocked=False):
        self.station = station
        self.target_station = target_station
        self.channel_eta_func = channel_eta_func
        super(SendOnScheduleEvent, self).__init__(time=time, offset_time=offset_time, required_objects=[station], priority=priority, ignore_blocked=ignore_blocked)

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, offset_time={self.offset_time}, station={self.station}, target_station={self.target_station}, channel_eta_func={self.channel_eta_func}, priority={self.priority}, ignore_blocked={self.ignore_blocked})"

    def _main_effect(self):
        for qubit in self.station.qubits:
            qubit.pair.qubits[0].update_time()
            qubit.pair.qubits[1].update_time()
            qubit.pair.update_time()
            send_distance = distance(qubit.station, self.target_station)
            arrive_time = self.event_queue.current_time + send_distance / C
            p_suc = self.channel_eta_func(send_distance)
            self.station.remove_qubit(qubit)
            qubit.station = None
            qubit.is_blocked = True
            if np.random.random() <= p_suc:
                event = QubitArrivesAtStationEvent(time=arrive_time, qubit=qubit, station=self.target_station)
                self.event_queue.add_event(event)
            else:
                event = DiscardQubitEvent(time=arrive_time, qubit=qubit)
                self.event_queue.add_event(event)
        return {"event_type": self.type, "resolve_successful": True}


class QubitArrivesAtStationEvent(Event):
    def __init__(self, time, qubit, station, priority=20, ignore_blocked=True):
        self.qubit = qubit
        self.station = station
        super(QubitArrivesAtStationEvent, self).__init__(time=time, required_objects=[qubit, station], priority=priority, ignore_blocked=ignore_blocked)

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, qubit={self.qubit}, station={self.station}, priority={self.priority}, ignore_blocked={self.ignore_blocked})"

    def _main_effect(self):
        self.qubit.pair.qubits[0].update_time()
        self.qubit.pair.qubits[1].update_time()
        self.qubit.pair.update_time()
        self.station.qubits += [self.qubit]
        self.qubit.station = self.station
        self.qubit.is_blocked = False
        if self.station.memory_cutoff_time is not None:
            discard_event = DiscardQubitEvent(time=self.event_queue.current_time + self.station.memory_cutoff_time, qubit=self.qubit)
            self.event_queue.add_event(discard_event)
        return {"event_type": self.type, "resolve_successful": True}


class UplinkSendProtocol(Protocol):
    def __init__(self, world, stations, sources, send_interval, channel_eta_func):
        self.time_list = []
        self.state_list = []
        self.resource_cost_max_list = []
        self.resource_cost_add_list = []
        self.stations = stations
        self.sources = sources
        self.send_interval = send_interval
        self.channel_eta_func = channel_eta_func
        super(UplinkSendProtocol, self).__init__(world=world)

    def setup(self):
        assert len(self.stations) == 5
        self.station_A = self.stations[0]
        self.sat_left = self.stations[1]
        self.sat_central = self.stations[2]
        self.sat_right = self.stations[3]
        self.station_B = self.stations[4]
        assert len(self.sources) == 2
        self.source_A, self.source_B = self.sources
        assert callable(getattr(self.source_A, "schedule_event", None))  # schedule_event is a required method for this protocol
        assert callable(getattr(self.source_B, "schedule_event", None))
        # now fire the repeating events
        left_event = SendOnScheduleEvent(time=self.world.event_queue.current_time,
                                         offset_time=self.send_interval,
                                         station=self.sat_left,
                                         target_station=self.sat_central,
                                         channel_eta_func=self.channel_eta_func)
        right_event = SendOnScheduleEvent(time=self.world.event_queue.current_time,
                                          offset_time=self.send_interval,
                                          station=self.sat_right,
                                          target_station=self.sat_central,
                                          channel_eta_func=self.channel_eta_func)
        self.world.event_queue.add_event(left_event)
        self.world.event_queue.add_event(right_event)

    def _eval_pair(self, long_range_pair):
        comm_distance = np.max([distance(self.sat_central, self.station_A), distance(self.station_B, self.sat_central)])
        comm_time = comm_distance / C

        self.time_list += [self.world.event_queue.current_time + comm_time]
        self.state_list += [long_range_pair.state]
        self.resource_cost_max_list += [long_range_pair.resource_cost_max]
        self.resource_cost_add_list += [long_range_pair.resource_cost_add]
        return

    def _left_pairs_scheduled(self):
        return list(filter(lambda event: (isinstance(event, SourceEvent)
                           and (self.station_A in event.source.target_stations)
                           and (self.sat_left in event.source.target_stations)
                           ),
                    self.world.event_queue.queue))

    def _right_pairs_scheduled(self):
        return list(filter(lambda event: (isinstance(event, SourceEvent)
                           and (self.sat_right in event.source.target_stations)
                           and (self.station_B in event.source.target_stations)
                           ),
                    self.world.event_queue.queue))

    def _left_pairs_swap_ready(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(filter(lambda pair: pair.is_between_stations(self.station_A, self.sat_central), pairs))

    def _rigth_pairs_swap_ready(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(filter(lambda pair: pair.is_between_stations(self.sat_central, self.station_B), pairs))

    def _get_long_range_pairs(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(filter(lambda pair: pair.is_between_stations(self.station_A, self.station_B), pairs))

    def check(self):
        if not self._left_pairs_scheduled():
            self.source_A.schedule_event()
        if not self._right_pairs_scheduled():
            self.source_B.schedule_event()

        if self.sat_central.qubits:
            left_pairs = self._left_pairs_swap_ready()
            num_left_pairs = len(left_pairs)
            right_pairs = self._right_pairs_swap_ready()
            num_right_pairs = len(right_pairs)
            if num_left_pairs != 0 and num_right_pairs != 0:
                num_swappings = min(num_left_pairs, num_right_pairs)
                for left_pair, right_pair in zip(left_pairs[:num_swappings], right_pairs[:num_swappings]):
                    # assert that we do not schedule the same swapping more than once
                    try:
                        next(filter(lambda event: (isinstance(event, EntanglementSwappingEvent)
                                                   and (left_pair in event.pairs)
                                                   and (right_pair in event.pairs)
                                                   ),
                                    self.world.event_queue.queue))
                        is_already_scheduled = True
                    except StopIteration:
                        is_already_scheduled = False
                    if not is_already_scheduled:
                        ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time, pairs=[left_pair, right_pair])
                        self.world.event_queue.add_event(ent_swap_event)

            long_range_pairs = self._get_long_range_pairs()
            if long_range_pairs:
                for long_range_pair in long_range_pairs:
                    self._eval_pair(long_range_pair)
                    # cleanup
                    long_range_pair.qubits[0].destroy()
                    long_range_pair.qubits[1].destroy()
                    long_range_pair.destroy()
                self.check()

        # protocol description:
        # uplinks try to establish pairs continously (check if nrp protocol also does this correctly)
        # at set times the two edge satellites will send all the pairs they have to the central station
        # Idea: do stuff that occurs on a clock with events that reschedule themselves?
        # Central station will do swapping if two qubits arrive at the same time. We need a mechanism to figure out what "at the same time" means.


def run(length, max_iter, params, send_interval, first_satellite_ground_dist_multiplier=0.25):
    try:
        P_LINK = params["P_LINK"]
    except KeyError:
        P_LINK = 1.0
    try:
        T_P = 1 / params["f_clock"]
    except KeyError:
        try:
            T_P = params["T_P"]  # preparation time
        except KeyError:
            T_P = 0
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError:
        T_DP = 1.0
    try:
        E_MA = params["E_MA"]  # misalignment error
    except KeyError:
        E_MA = 0
    try:
        P_D = params["P_D"]
    except KeyError:
        P_D = 0
    try:
        LAMBDA_BSM = params["LAMBDA_BSM"]
    except KeyError:
        LAMBDA_BSM = 1
    try:
        ORBITAL_HEIGHT = params["ORBITAL_HEIGHT"]
    except KeyError as e:
        raise Exception('params["ORBITAL_HEIGHT"] is a mandatory argument').with_traceback(e.__traceback__)
    try:
        SENDER_APERTURE_RADIUS = params["SENDER_APERTURE_RADIUS"]
    except KeyError as e:
        raise Exception('params["SENDER_APERTURE_RADIUS"] is a mandatory argument').with_traceback(e.__traceback__)
    try:
        RECEIVER_APERTURE_RADIUS = params["RECEIVER_APERTURE_RADIUS"]
    except KeyError as e:
        raise Exception('params["RECEIVER_APERTURE_RADIUS"] is a mandatory argument').with_traceback(e.__traceback__)
    try:
        DIVERGENCE_THETA = params["DIVERGENCE_THETA"]
    except KeyError as e:
        raise Exception('params["DIVERGENCE_THETA"] is a mandatory argument').with_traceback(e.__traceback__)

    def position_from_angle(radius, angle):
        return radius * np.array([np.sin(angle), np.cos(angle)])

    station_a_angle = 0
    station_a_position = position_from_angle(R_E, station_a_angle)
    first_satellite_angle = first_satellite_ground_dist_multiplier * length / R_E
    first_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, first_satellite_angle)
    second_satellite_angle = length / 2 / R_E
    second_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, second_satellite_angle)
    third_satellite_angle = (1 - first_satellite_ground_dist_multiplier) * length / R_E
    third_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, third_satellite_angle)
    station_b_angle = length / R_E
    station_b_position = position_from_angle(R_E, station_b_angle)
    elevation_left = elevation_curved(ground_dist=first_satellite_ground_dist_multiplier * length, h=ORBITAL_HEIGHT)
    arrival_chance_left = eta_atm(elevation_left) \
                          * eta_dif(distance=distance(station_a_position, first_satellite_position),
                                    divergence_half_angle=DIVERGENCE_THETA,
                                    sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                    receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)
    elevation_right = elevation_curved(ground_dist=first_satellite_ground_dist_multiplier * length, h=ORBITAL_HEIGHT)
    arrival_chance_right = eta_atm(elevation_right) \
                           * eta_dif(distance=distance(station_b_position, third_satellite_position),
                                     divergence_half_angle=DIVERGENCE_THETA,
                                     sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                     receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)

    def eta_channel_func(dist):
        return eta_dif(distance=dist,
                       divergence_half_angle=DIVERGENCE_THETA,
                       sender_aperture_radius=SENDER_APERTURE_RADIUS,
                       receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1 - LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

    def generate_time_distribution(arrival_chance):
        def time_distribution(source):
            trial_time = T_P
            eta = P_LINK * arrival_chance
            eta_effective = 1 - (1 - eta) * (1 - P_D)**2
            random_num = np.random.geometric(eta_effective)
            return random_num * trial_time, random_num
        return time_distribution

    @lru_cache()
    def state_generation_left(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        ground_station = source.target_stations[0]
        # note how no additional dephasing occurs here in this scenari
        if ground_station.dark_count_probability is not None:
            eta = P_LINK * arrival_chance_left
            state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=0, rho=state, alpha=alpha_of_eta(eta=eta, p_d=ground_station.dark_count_probability))
        return state

    @lru_cache()
    def state_generation_right(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        ground_station = source.target_stations[1]
        # note how no additional dephasing occurs here in this scenari
        if ground_station.dark_count_probability is not None:
            eta = P_LINK * arrival_chance_right
            state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=0, rho=state, alpha=alpha_of_eta(eta=eta, p_d=ground_station.dark_count_probability))
        return state

    misalignment_noise = NoiseChannel(n_qubits=1, channel_function=construct_y_noise_channel(epsilon=E_MA))

    # do setup
    world = World()
    station_A = Station(world, position=station_a_position, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )
    sat_left = Station(world, position=first_satellite_position, memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP))
    sat_central = Station(world, position=second_satellite_position,
                          memory_noise=None,
                          memory_cutoff_time=1e-6,
                          BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                          )
    sat_right = Station(world, position=third_satellite_position, memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP))
    station_B = Station(world, position=station_b_position, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )

    source_A = SchedulingSource(world, position=first_satellite_position, target_stations=[station_A, sat_left], time_distribution=generate_time_distribution(arrival_chance_left), state_generation=state_generation_left)
    source_B = SchedulingSource(world, position=third_satellite_position, target_stations=[sat_right, station_B], time_distribution=generate_time_distribution(arrival_chance_right), state_generation=state_generation_right)
    protocol = UplinkSendProtocol(world, stations=[station_A, sat_left, sat_central, sat_right, station_B], sources=[source_A, source_B], send_interval=send_interval, channel_eta_func=eta_channel_func)
    protocol.setup()  # this also starts the repeating send events off

    while world.event_queue.current_time < 0.049999:
        protocol.check()
        world.event_queue.resolve_next_event()

    import code
    code.interact(local=locals())

    while len(protocol.time_list) < max_iter:
        protocol.check()
        world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":
    test_params = {"P_LINK": 0.56, "f_clock": 1e3, "T_DP": 0.1, "P_D": 10**-6, "ORBITAL_HEIGHT": 400e3, "SENDER_APERTURE_RADIUS": 0.15, "RECEIVER_APERTURE_RADIUS": 0.50, "DIVERGENCE_THETA": 5e-6}
    p = run(length=400e3, max_iter=10, params=test_params, send_interval=0.05, first_satellite_ground_dist_multiplier=0.25)
