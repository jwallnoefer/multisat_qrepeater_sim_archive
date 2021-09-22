import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from world import World
from protocol import TwoLinkProtocol
from events import EntanglementSwappingEvent
import libs.matrix as mat
from libs.aux_functions import apply_single_qubit_map, y_noise_channel, z_noise_channel, w_noise_channel, distance
from consts import ETA_ATM_PI_HALF_780_NM
from consts import AVERAGE_EARTH_RADIUS as R_E
from consts import SPEED_OF_LIGHT_IN_VACCUM as C
from functools import lru_cache
from noise import NoiseModel, NoiseChannel
from quantum_objects import SchedulingSource, Station


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


class MultiMemoryProtocol(TwoLinkProtocol):
    def __init__(self, world, num_memories):
        self.num_memories = num_memories
        super(MultiMemoryProtocol, self).__init__(world=world)

    def check(self):
        left_pairs = self._get_left_pairs()
        num_left_pairs = len(left_pairs)
        right_pairs = self._get_right_pairs()
        num_right_pairs = len(right_pairs)
        num_left_pairs_scheduled = len(self._left_pairs_scheduled())
        num_right_pairs_scheduled = len(self._right_pairs_scheduled())
        left_used = num_left_pairs + num_left_pairs_scheduled
        right_used = num_right_pairs + num_right_pairs_scheduled

        if left_used < self.num_memories:
            for _ in range(self.num_memories - left_used):
                self.source_A.schedule_event()
        if right_used < self.num_memories:
            for _ in range(self.num_memories - right_used):
                self.source_B.schedule_event()

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


def run(length, max_iter, params, cutoff_time=None, num_memories=1, first_satellite_ground_dist_multiplier=0.25, return_world=False):
    # unpack the parameters
    # print(length)
    try:
        P_LINK = params["P_LINK"]
    except KeyError:
        P_LINK = 1.0
    try:
        ETA_DET = params["ETA_DET"]
    except KeyError as e:
        raise Exception('params["ETA_DET"] is a mandatory argument').with_traceback(e.__traceback__)
    try:
        ETA_MEM = params["ETA_MEM"]
    except KeyError as e:
        raise Exception('params["ETA_MEM"] is a mandatory argument').with_traceback(e.__traceback__)
    try:
        F_CLOCK = params["F_CLOCK"]
    except KeyError as e:
        raise Exception('params["F_CLOCK"] is a mandatory argument').with_traceback(e.__traceback__)
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
    first_satellite_angle = first_satellite_ground_dist_multiplier * length / R_E
    first_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, first_satellite_angle)
    second_satellite_angle = length / 2 / R_E
    second_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, second_satellite_angle)
    third_satellite_angle = (1 - first_satellite_ground_dist_multiplier) * length / R_E
    third_satellite_position = position_from_angle(R_E + ORBITAL_HEIGHT, third_satellite_angle)
    station_b_angle = length / R_E
    station_b_position = position_from_angle(R_E, station_b_angle)
    elevation_left = elevation_curved(ground_dist=first_satellite_ground_dist_multiplier * length, h=ORBITAL_HEIGHT)
    arrival_chance_a_left = eta_atm(elevation_left) \
                            * eta_dif(distance=distance(station_a_position, first_satellite_position),
                                      divergence_half_angle=DIVERGENCE_THETA,
                                      sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                      receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)
    arrival_chance_left_center = eta_dif(distance=distance(first_satellite_position, second_satellite_position),
                                         divergence_half_angle=DIVERGENCE_THETA,
                                         sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                         receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)
    arrival_chance_left = arrival_chance_a_left * arrival_chance_left_center
    elevation_right = elevation_curved(ground_dist=first_satellite_ground_dist_multiplier * length, h=ORBITAL_HEIGHT)
    arrival_chance_b_right = eta_atm(elevation_right) \
                             * eta_dif(distance=distance(station_b_position, third_satellite_position),
                                       divergence_half_angle=DIVERGENCE_THETA,
                                       sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                       receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)
    arrival_chance_right_center = eta_dif(distance=distance(third_satellite_position, second_satellite_position),
                                          divergence_half_angle=DIVERGENCE_THETA,
                                          sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                          receiver_aperture_radius=RECEIVER_APERTURE_RADIUS)
    arrival_chance_right = arrival_chance_b_right * arrival_chance_right_center

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1 - LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

    def time_distribution_left(source):
        ground_station = source.target_stations[0]  # here the first station is the ground station
        ground_station_distance = distance(source, ground_station)
        ground_station_time = ground_station_distance / C
        satellite = source.target_stations[1]
        satellite_distance = distance(source, satellite)
        satellite_time = satellite_distance / C
        comm_distance = distance(source.target_stations[0], source.target_stations[1])  # ground station needs to communicate with middle satellite because that is where the memories are
        # careful: doesn't account for the case where memory satellite is below the horizon
        comm_time = comm_distance / C
        satellite_eta = ETA_MEM * arrival_chance_left_center
        ground_eta = ETA_DET * arrival_chance_a_left
        ground_eta_effective = 1 - (1 - ground_eta) * (1 - P_D)**2
        # now do the calculation for inner and outer state generation loop
        inner_time = num_memories * 1 / F_CLOCK
        outer_time = ground_station_time - satellite_time + comm_time
        num_outer = np.random.geometric(ground_eta_effective)
        num_inner = 0
        for _ in range(num_outer):
            num_inner += np.random.geometric(satellite_eta)
        total_time = num_inner * inner_time + num_outer * outer_time
        total_trials = num_inner
        return total_time, total_trials

    def time_distribution_right(source):
        ground_station = source.target_stations[1]  # here the first station is the ground station
        ground_station_distance = distance(source, ground_station)
        ground_station_time = ground_station_distance / C
        satellite = source.target_stations[0]
        satellite_distance = distance(source, satellite)
        satellite_time = satellite_distance / C
        comm_distance = distance(source.target_stations[0], source.target_stations[1])  # ground station needs to communicate with middle satellite because that is where the memories are
        # careful: doesn't account for the case where memory satellite is below the horizon
        comm_time = comm_distance / C
        satellite_eta = ETA_MEM * arrival_chance_left_center
        ground_eta = ETA_DET * arrival_chance_a_left
        ground_eta_effective = 1 - (1 - ground_eta) * (1 - P_D)**2
        # now do the calculation for inner and outer state generation loop
        inner_time = num_memories * 1 / F_CLOCK
        outer_time = ground_station_time - satellite_time + comm_time
        num_outer = np.random.geometric(ground_eta_effective)
        num_inner = 0
        for _ in range(num_outer):
            num_inner += np.random.geometric(satellite_eta)
        total_time = num_inner * inner_time + num_outer * outer_time
        total_trials = num_inner
        return total_time, total_trials

    @lru_cache()
    def state_generation_left(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        ground_station = source.target_stations[0]  # here the first station is the ground station
        ground_station_distance = distance(source, ground_station)
        ground_station_time = ground_station_distance / C
        satellite = source.target_stations[1]
        satellite_distance = distance(source, satellite)
        satellite_time = satellite_distance / C
        comm_distance = distance(ground_station, satellite)  # ground station needs to communicate with middle satellite because that is where the memories are
        # careful: doesn't account for the case where memory satellite is below the horizon
        comm_time = comm_distance / C
        storage_time = ground_station_time + comm_time - satellite_time
        if satellite.memory_noise is not None:
            state = apply_single_qubit_map(map_func=satellite.memory_noise, qubit_index=1, rho=state, t=storage_time)
        if ground_station.dark_count_probability is not None:
            eta = P_LINK * arrival_chance_left
            state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=0, rho=state, alpha=alpha_of_eta(eta=eta, p_d=ground_station.dark_count_probability))
        return state

    @lru_cache()
    def state_generation_right(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        ground_station = source.target_stations[1]  # here the second station is the ground station
        ground_station_distance = distance(source, ground_station)
        ground_station_time = ground_station_distance / C
        satellite = source.target_stations[0]
        satellite_distance = distance(source, satellite)
        satellite_time = satellite_distance / C
        comm_distance = distance(ground_station, satellite)  # ground station needs to communicate with middle satellite because that is where the memories are
        # careful: doesn't account for the case where memory satellite is below the horizon
        comm_time = comm_distance / C
        storage_time = ground_station_time + comm_time - satellite_time
        if satellite.memory_noise is not None:
            state = apply_single_qubit_map(map_func=satellite.memory_noise, qubit_index=0, rho=state, t=storage_time)
        if ground_station.dark_count_probability is not None:
            eta = P_LINK * arrival_chance_right
            state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=1, rho=state, alpha=alpha_of_eta(eta=eta, p_d=ground_station.dark_count_probability))
        return state

    misalignment_noise = NoiseChannel(n_qubits=1, channel_function=construct_y_noise_channel(epsilon=E_MA))

    world = World()
    station_A = Station(world, position=station_a_position, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )
    station_central = Station(world, position=second_satellite_position,
                              memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP),
                              memory_cutoff_time=cutoff_time,
                              BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                              )
    station_B = Station(world, position=station_b_position, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )

    source_A = SchedulingSource(world, position=first_satellite_position, target_stations=[station_A, station_central], time_distribution=time_distribution_left, state_generation=state_generation_left)
    source_B = SchedulingSource(world, position=third_satellite_position, target_stations=[station_central, station_B], time_distribution=time_distribution_right, state_generation=state_generation_right)
    protocol = MultiMemoryProtocol(world, num_memories=num_memories)
    protocol.setup()

    # import code
    # code.interact(local=locals())

    while len(protocol.time_list) < max_iter:
        protocol.check()
        world.event_queue.resolve_next_event()

    if return_world:
        return protocol, world
    else:
        return protocol


if __name__ == "__main__":
    length_list = np.linspace(0e3, 8000e3, num=16)
    from time import time
    for length in length_list:
        start_time = time()
        p, w = run(length=length, max_iter=100,
                   params={"P_LINK": 0.56, "ETA_MEM": 0.8, "ETA_DET": 0.7, "F_CLOCK": 1 / 20e6,
                           "T_DP": 0.1, "P_D": 10**-6, "ORBITAL_HEIGHT": 400e3, "SENDER_APERTURE_RADIUS": 0.15,
                           "RECEIVER_APERTURE_RADIUS": 0.50, "DIVERGENCE_THETA": 2e-6},
                   cutoff_time=0.05, num_memories=10, first_satellite_ground_dist_multiplier=0, return_world=True)
        print(f"{length=} finished in {(time()-start_time):.2f} seconds.")
        w.event_queue.print_stats()
    # import matplotlib.pyplot as plt
    # length_list = np.linspace(0e3, 8000e3, num=10)
    # # length_list = [0]
    # for sat_pos in [0, 0.125, 0.25, 0.375, 0.5]:
    #     print(sat_pos)
    #     # ps = [run(length=length, max_iter=1000, params={"P_LINK": 0.56, "T_DP": 1, "P_D": 10**-6, "ORBITAL_HEIGHT": 400e3, "SENDER_APERTURE_RADIUS": 0.15, "RECEIVER_APERTURE_RADIUS": 0.50, "DIVERGENCE_THETA": 10e-6}, cutoff_time=0.5, num_memories=1000, first_satellite_ground_dist_multiplier=sat_pos) for length in length_list]
    #     ps = [run(length=length, max_iter=100, params={"P_LINK": 0.56, "ETA_MEM": 0.8, "ETA_DET": 0.7, "T_DP": 1, "P_D": 10**-6, "ORBITAL_HEIGHT": 400e3, "SENDER_APERTURE_RADIUS": 0.15, "RECEIVER_APERTURE_RADIUS": 0.50, "DIVERGENCE_THETA": 2e-6}, cutoff_time=0.5, num_memories=1000, first_satellite_ground_dist_multiplier=sat_pos) for length in length_list]
    #     from libs.aux_functions import standard_bipartite_evaluation
    #     res = [standard_bipartite_evaluation(p.data) for p in ps]
    #     plt.errorbar(length_list / 1000, [r[4] / 2 for r in res], yerr=[r[5] / 2 for r in res], fmt="o", label=str(sat_pos))  # 10 * np.log10(key_per_resource))
    # plt.yscale("log")
    # plt.legend()
    # plt.show()
