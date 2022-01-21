import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from world import World
from protocol import Protocol, MessageReadingProtocol
from events import SourceEvent, EntanglementSwappingEvent
import libs.matrix as mat
from libs.aux_functions import apply_single_qubit_map, y_noise_channel, z_noise_channel, w_noise_channel, distance
from scenarios.three_satellites.common_functions import alpha_of_eta, construct_dephasing_noise_channel, construct_y_noise_channel, eta_dif, eta_atm, elevation_curved
from consts import ETA_ATM_PI_HALF_780_NM
from consts import AVERAGE_EARTH_RADIUS as R_E
from consts import SPEED_OF_LIGHT_IN_VACCUM as C
from functools import lru_cache
from noise import NoiseModel, NoiseChannel
from quantum_objects import SchedulingSource, Station
from functools import lru_cache
from collections import defaultdict
import pandas as pd
from warnings import warn


@lru_cache(maxsize=int(1e6))
def is_event_swapping_pairs(event, pair1, pair2):
    return isinstance(event, EntanglementSwappingEvent) and (pair1 in event.pairs) and (pair2 in event.pairs)


@lru_cache(maxsize=int(1e6))
def is_sourceevent_between_stations(event, station1, station2):
    return isinstance(event, SourceEvent) and (station1 in event.source.target_stations) and (station2 in event.source.target_stations)


class FourlinkProtocol(MessageReadingProtocol):
    #
    def __init__(self, world, num_memories, stations, sources):
        self.num_memories = num_memories
        self.time_list = []
        self.state_list = []
        self.resource_cost_max_list = []
        self.resource_cost_add_list = []
        self.stations = stations
        self.sources = sources
        self.scheduled_swappings = defaultdict(lambda: [])
        super(FourlinkProtocol, self).__init__(world=world)

    def setup(self):
        # Station ordering left to right
        assert len(self.stations) == 5
        self.station_ground_left = self.stations[0]
        self.sat_left = self.stations[1]
        self.sat_central = self.stations[2]
        self.sat_right = self.stations[3]
        self.station_ground_right = self.stations[4]
        assert len(self.sources) == 4
        self.source_link1, self.source_link2, self.source_link3, self.source_link4 = self.sources
        for source in self.sources:
            assert callable(getattr(source, "schedule_event", None))# schedule_event is a required method for this protocol
        self.link_stations = [[self.stations[i], self.stations[i+1]] for i in range(4)]

    @property
    def data(self):
        return pd.DataFrame({"time": self.time_list, "state": self.state_list,
                             "resource_cost_max": self.resource_cost_max_list,
                             "resource_cost_add": self.resource_cost_add_list})

    def _get_pairs_between_stations(self, station1, station2):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(filter(lambda pair: pair.is_between_stations(station1, station2), pairs))

    def _get_pairs_scheduled(self, station1, station2):
        return list(filter(lambda event: is_sourceevent_between_stations(event, station1, station2),
                    self.world.event_queue.queue))

    def _eval_pair(self, long_range_pair):
        comm_distance = np.max([distance(self.sat_central, self.station_ground_left), distance(self.sat_central, self.station_ground_right)])
        comm_time = comm_distance / C

        self.time_list += [self.world.event_queue.current_time + comm_time]
        self.state_list += [long_range_pair.state]
        self.resource_cost_max_list += [long_range_pair.resource_cost_max]
        self.resource_cost_add_list += [long_range_pair.resource_cost_add]
        return

    def _check_middle_station_overflow(self):
        # first check if middle station memories are too full:
        left_pairs, right_pairs = self.pairs_at_station(self.sat_central)
        has_overflowed = False
        if len(left_pairs) > self.num_memories:
            last_pair = left_pairs[-1]
            last_pair.qubits[0].destroy()
            last_pair.qubits[1].destroy()
            last_pair.destroy_and_track_resources()
            has_overflowed = True
        if len(right_pairs) > self.num_memories:
            last_pair = right_pairs[-1]
            last_pair.qubits[0].destroy()
            last_pair.qubits[1].destroy()
            last_pair.destroy_and_track_resources()
            has_overflowed = True
        return has_overflowed

    def _check_new_source_events(self):
        free_left_memories = self.memory_check(self.sat_left)
        free_right_memories = self.memory_check(self.sat_right)
        for free_memories, source in zip(free_left_memories + free_right_memories, self.sources):
            for _ in range(free_memories):
                source.schedule_event()

    def _check_swapping(self):
        #Swapping loop
        for station in self.stations[1:-1]:
            left_pairs, right_pairs = self.pairs_at_station(station)
            num_swappings = min(len(left_pairs), len(right_pairs))
            if num_swappings:
                # get rid of events that are no longer scheduled
                self.scheduled_swappings[station] = [event for event in self.scheduled_swappings[station] if event in self.world.event_queue.queue]
            for left_pair, right_pair in zip(left_pairs[:num_swappings], right_pairs[:num_swappings]):
                # assert that we do not schedule the same swapping more than once
                try:
                    next(filter(lambda event: is_event_swapping_pairs(event, left_pair, right_pair), self.scheduled_swappings[station]))
                    is_already_scheduled = True
                except StopIteration:
                    is_already_scheduled = False
                if not is_already_scheduled:
                    ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time, pairs=[left_pair, right_pair])
                    self.scheduled_swappings[station] += [ent_swap_event]
                    self.world.event_queue.add_event(ent_swap_event)

    def _check_long_distance_pair(self):
        #Evaluate long range pairs
        long_range_pairs = self._get_pairs_between_stations(self.station_ground_left, self.station_ground_right)
        if long_range_pairs:
            for long_range_pair in long_range_pairs:
                self._eval_pair(long_range_pair)
                # cleanup
                long_range_pair.qubits[0].destroy()
                long_range_pair.qubits[1].destroy()
                long_range_pair.destroy()
            # self.check()  # was useful at some point for other scenarios

    def check(self, message=None):
        if message is None:
            self._check_middle_station_overflow()
            self._check_new_source_events()
            self._check_swapping()
            self._check_long_distance_pair()
        elif message["event_type"] == "SourceEvent" and message["resolve_successful"] is True:
            has_overflowed = self._check_middle_station_overflow()
            if has_overflowed:
                self._check_new_source_events()
            self._check_swapping()
        elif message["event_type"] == "DiscardQubitEvent" and message["resolve_successful"] is True:
            self._check_new_source_events()
        elif message["event_type"] == "DiscardQubitEvent" and message["resolve_successful"] is False:
            pass
        elif message["event_type"] == "EntanglementSwappingEvent" and message["resolve_successful"] is True:
            self._check_new_source_events()
            self._check_long_distance_pair()
        elif message["event_type"] == "EntanglementSwappingEvent" and message["resolve_successful"] is False:
            self._check_swapping()
        elif message["event_type"] == "SourceEvent" and message["resolve_successful"] is False:
            warn("A SourceEvent has resolved unsuccessfully. This should never happen.")
        else:
            warn(f"Unrecognized message type encountered: {message}")


    def pairs_at_station(self, station):
        station_index = self.stations.index(station)
        pairs_left = []
        pairs_right = []
        for qubit in station.qubits:
            pair = qubit.pair
            qubit_list = list(pair.qubits)
            qubit_list.remove(qubit)
            qubit_neighbor = qubit_list[0]
            if self.stations.index(qubit_neighbor.station) < station_index:
                pairs_left += [pair]
            else:
                pairs_right += [pair]
        return (pairs_left, pairs_right)


    def memory_check(self, station):
        station_index = self.stations.index(station)
        free_memories_left = self.num_memories
        free_memories_right = self.num_memories
        pairs_left, pairs_right = self.pairs_at_station(station)
        free_memories_left -= len(pairs_left)
        free_memories_right -= len(pairs_right)
        free_memories_left -= len(self._get_pairs_scheduled(self.stations[station_index-1], station))
        free_memories_right -= len(self._get_pairs_scheduled(station, self.stations[station_index+1]))
        return (free_memories_left, free_memories_right)

    def memory_check_global(self):
        free_memories = {station: self.memory_check(station) for station in self.stations[1:-1]}
        free_memories[self.station_ground_left] = (self.num_memories, self.num_memories)
        free_memories[self.station_ground_right] = (self.num_memories, self.num_memories)
        return free_memories


def run(length, max_iter, params, cutoff_time=None, num_memories=2, first_satellite_ground_dist_multiplier=None, satellite_multipliers=None, return_world=False):
    if first_satellite_ground_dist_multiplier is None and satellite_multipliers is None:
        raise ValueError("Must specify either first_satellite_ground_dist_multiplier or satellite_multipliers.")
    elif first_satellite_ground_dist_multiplier is not None and satellite_multipliers is not None:
        raise ValueError(f"Only one of first_satellite_ground_dist_multiplier or satellite_multipliers can be specified. run was called with {first_satellite_ground_dist_multiplier=}, {satellite_multipliers=}")

    # unpack the parameters
    # print(length)
    try:
        ETA_DET = params["ETA_DET"]
    except KeyError:
        ETA_DET = 1.0
    try:
        ETA_MEM = params["ETA_MEM"]
    except KeyError:
        ETA_MEM = 1.0
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
        P_D = params["P_D"]  # dark count probability
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
    if satellite_multipliers is not None:
        first_satellite_angle, second_satellite_angle, third_satellite_angle = np.array(satellite_multipliers) * length / R_E
    else:
        first_satellite_angle = first_satellite_ground_dist_multiplier * length / R_E
        second_satellite_angle = length / 2 / R_E
        third_satellite_angle = (1 - first_satellite_ground_dist_multiplier) * length / R_E
    first_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, first_satellite_angle)
    second_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, second_satellite_angle)
    third_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, third_satellite_angle)
    station_b_angle = length / R_E
    station_b_position = position_from_angle(R_E, station_b_angle)
    elevation_left = elevation_curved(ground_dist=np.abs(station_a_angle - first_satellite_angle) * R_E, h=ORBITAL_HEIGHT)
    elevation_right = elevation_curved(ground_dist=np.abs(station_b_angle - third_satellite_angle) * R_E, h=ORBITAL_HEIGHT)
    arrival_chance_link1 = eta_atm(elevation_left) \
                          * eta_dif(distance=distance(station_a_position, first_satellite_position),
                                    divergence_half_angle=DIVERGENCE_THETA,
                                    sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                    receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)
    arrival_chance_link2 = eta_dif(distance=distance(first_satellite_position, second_satellite_position),
                                   divergence_half_angle=DIVERGENCE_THETA,
                                   sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                   receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)
    arrival_chance_link3 = eta_dif(distance=distance(second_satellite_position, third_satellite_position),
                                   divergence_half_angle=DIVERGENCE_THETA,
                                   sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                   receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)
    arrival_chance_link4 = eta_atm(elevation_right) \
                          * eta_dif(distance=distance(third_satellite_position, station_b_position),
                                    divergence_half_angle=DIVERGENCE_THETA,
                                    sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                    receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)

    def generate_time_distribution(arrival_chance, p_link):
        def time_distribution(source):
            comm_distance = np.max([distance(source, source.target_stations[0]), distance(source.target_stations[1], source)])
            comm_time = 2 * comm_distance / C
            eta = p_link * arrival_chance
            eta_effective = 1 - (1 - eta) * (1 - P_D)**2
            trial_time = T_P + comm_time  # I don't think that paper uses latency time and loading time?
            random_num = np.random.geometric(eta_effective)
            return random_num * trial_time, random_num
        return time_distribution

    time_distribution_link1 = generate_time_distribution(arrival_chance_link1, ETA_MEM*ETA_DET)
    time_distribution_link2 = generate_time_distribution(arrival_chance_link2, ETA_MEM**2)
    time_distribution_link3 = generate_time_distribution(arrival_chance_link3, ETA_MEM**2)
    time_distribution_link4 = generate_time_distribution(arrival_chance_link4, ETA_MEM*ETA_DET)

    def generate_state_generation(arrival_chance, p_link):
        @lru_cache()
        def state_generation(source):
            state = np.dot(mat.phiplus, mat.H(mat.phiplus))
            comm_distance = np.max([distance(source, source.target_stations[0]), distance(source.target_stations[1], source)])
            trial_time = 2 * comm_distance / C
            for idx, station in enumerate(source.target_stations):
                if station.memory_noise is not None:  # dephasing that has accrued during trial
                    storage_time = trial_time - distance(source, station) / C  # qubit is in storage for varying amounts of time
                    state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
                if station.dark_count_probability is not None:  # dark counts are handled here because the information about eta is needed for that
                    eta = p_link * arrival_chance
                    state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability))
            return state
        return state_generation

    state_generation_link1 = generate_state_generation(arrival_chance_link1, ETA_MEM*ETA_DET)
    state_generation_link2 = generate_state_generation(arrival_chance_link2, ETA_MEM**2)
    state_generation_link3 = generate_state_generation(arrival_chance_link3, ETA_MEM**2)
    state_generation_link4 = generate_state_generation(arrival_chance_link4, ETA_MEM*ETA_DET)

    misalignment_noise = NoiseChannel(n_qubits=1, channel_function=construct_y_noise_channel(epsilon=E_MA))

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1 - LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])


    world = World()

    station_ground_left = Station(world, position = station_a_position,
                                  dark_count_probability=P_D,
                                  creation_noise_channel=misalignment_noise)
    station_sat_left = Station(world, position = first_satellite_position,
                               memory_cutoff_time=cutoff_time,
                               memory_noise = construct_dephasing_noise_channel(T_DP),
                               BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                              )
    station_sat_central = Station(world, position = second_satellite_position,
                                  memory_cutoff_time=cutoff_time,
                                  memory_noise = construct_dephasing_noise_channel(T_DP),
                                  BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                                 )
    station_sat_right = Station(world, position = third_satellite_position,
                                memory_cutoff_time=cutoff_time,
                                memory_noise = construct_dephasing_noise_channel(T_DP),
                                BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                               )
    station_ground_right = Station(world, position = station_b_position,
                                   dark_count_probability=P_D,
                                   creation_noise_channel=misalignment_noise)

    source_sat_left1 = SchedulingSource(world, position = first_satellite_position,
                                        target_stations = [station_ground_left, station_sat_left],
                                        time_distribution = time_distribution_link1,
                                        state_generation = state_generation_link1)
    source_sat_left2 = SchedulingSource(world, position = first_satellite_position,
                                        target_stations = [station_sat_left, station_sat_central],
                                        time_distribution = time_distribution_link2,
                                        state_generation = state_generation_link2)
    source_sat_right1 = SchedulingSource(world, position = third_satellite_position,
                                         target_stations = [station_sat_central, station_sat_right],
                                         time_distribution = time_distribution_link3,
                                         state_generation = state_generation_link3)
    source_sat_right2 = SchedulingSource(world, position = third_satellite_position,
                                         target_stations = [station_sat_right, station_ground_right],
                                         time_distribution = time_distribution_link4,
                                         state_generation = state_generation_link4)

    protocol = FourlinkProtocol(world, num_memories, stations = [station_ground_left, station_sat_left, station_sat_central, station_sat_right, station_ground_right], sources = [source_sat_left1 , source_sat_left2, source_sat_right1, source_sat_right2])
    protocol.setup()

    # import code
    # code.interact(local=locals())
    message = None
    while len(protocol.time_list) < max_iter:
        protocol.check(message)
        message = world.event_queue.resolve_next_event()

    if return_world:
        return protocol, world
    else:
        return protocol

if __name__ == "__main__":
    np.random.seed(48215485)
    from time import time
    start_time = time()
    run(length=250e3, max_iter=1, params={"ETA_MEM": 0.8, "ETA_DET": 0.7, "P_D": 10**-6, "T_DP": 0.1, "ORBITAL_HEIGHT": 400e3, "SENDER_APERTURE_RADIUS": 0.15, "RECEIVER_APERTURE_RADIUS": 0.50, "DIVERGENCE_THETA": 2e-6}, cutoff_time=0.01, num_memories=1000, first_satellite_ground_dist_multiplier=0)
    print(f"This took {(time() - start_time):.2f} seconds.")
    # # orbital_heights = [400e3, 1250e3, 4000e3, 12000e3, 36000e3]
    # orbital_heights = [300e3, 400e3, 500e3, 600e3, 800e3, 1250e3, 1500e3, 2000e3]
    # horizons = [2 * np.arccos(R_E/(R_E + orbit_height))*R_E  for orbit_height in orbital_heights]
    # # print([horizon/1e3 for horizon in horizons])
    # idx = -1
    # threshold = 10e-1
    # # lengths = np.linspace(100e3, horizons[idx], num=16)
    # lengths= np.linspace(250e3, 4000e3, num=20)
    # # lengths = [horizons[-1]]
    # cutoff_times = [0.1, 0.2, 0.3, 0.4, 0.5]
    # from time import time
    # from libs.aux_functions import standard_bipartite_evaluation
    # keys = {}
    # end_times = {}
    # for cutoff_time in cutoff_times:
    #     keys[cutoff_time] = []
    #     end_times[cutoff_time] = []
    #     for length in lengths:
    #         start_time = time()
    #         p, w = run(length=length, max_iter=100, params={"ETA_MEM": 0.8, "ETA_DET": 0.7, "P_D": 10**-6, "ORBITAL_HEIGHT": orbital_heights[idx], "SENDER_APERTURE_RADIUS": 0.15, "RECEIVER_APERTURE_RADIUS": 0.50, "DIVERGENCE_THETA": 2e-6}, cutoff_time=cutoff_time, num_memories=20, first_satellite_ground_dist_multiplier=0, return_world=True)
    #         df = p.data
    #         keys[cutoff_time] += [standard_bipartite_evaluation(df)[2]]
    #         end_times[cutoff_time] += [p.time_list[-1]]
    #         # w.event_queue.print_stats()
    #         # if keys[-1] < threshold:
    #         #     keys[-1] = 0
    #         print(f"length={length} finished after {(time()-start_time):.2f} seconds.")
    #
    # import matplotlib.pyplot as plt
    # for cutoff_time in cutoff_times:
    #     plt.plot(lengths, keys[cutoff_time], label=f"{cutoff_time:.2f}")
    # plt.grid()
    # plt.yscale("log")
    # plt.legend()
    # # plt.ylim(0, 1)
    # plt.title(f"Fourlink with orbital height {orbital_heights[idx]/(1e3)}km, theta = 2e-6rad, #memories = 20")
    # plt.xlabel(f"Distance (max {lengths[-1]/(1e3)}km)")
    # plt.ylabel(f"Keyrate")
    # plt.show()
    #
    # for cutoff_time in cutoff_times:
    #     plt.plot(lengths, end_times[cutoff_time], label=f"{cutoff_time:.2f}")
    # plt.legend()
    # plt.grid()
    # plt.title(f"Fourlink with orbital height {orbital_heights[idx]/(1e3)}km, theta = 2e-6rad, #memories = 20")
    # plt.xlabel(f"Distance (max {lengths[-1]/(1e3)}km)")
    # plt.ylabel(f"Time after 240 long distance pairs.")
    # plt.show()
