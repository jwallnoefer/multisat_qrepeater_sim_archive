import os, sys; sys.path.insert(0, os.path.abspath("."))
from quantum_objects import SchedulingSource, Station
from protocol import TwoLinkProtocol
from world import World
from events import EntanglementSwappingEvent
import libs.matrix as mat
import numpy as np
from libs.aux_functions import apply_single_qubit_map, w_noise_channel, distance
from scenarios.three_satellites.common_functions import alpha_of_eta, construct_dephasing_noise_channel, construct_y_noise_channel, eta_dif, eta_atm, elevation_curved
from noise import NoiseModel, NoiseChannel
from warnings import warn
from consts import AVERAGE_EARTH_RADIUS as R_E
from consts import SPEED_OF_LIGHT_IN_VACCUM as C
from functools import lru_cache


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


def run(length, max_iter, params, cutoff_time=None, num_memories=1,
        position_multiplier=0.5, mode="sim", return_world=False):
    # unpack the parameters
    try:
        P_LINK = params["P_LINK"]
    except KeyError:
        P_LINK = 1.0
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
    try:
        POINTING_ERROR_SIGMA = params["POINTING_ERROR_SIGMA"]
    except KeyError:
        warn('params["POINTING_ERROR_SIGMA"] is not defined, assuming zero.')
        POINTING_ERROR_SIGMA = 0

    def position_from_angle(radius, angle):
        return radius * np.array([np.sin(angle), np.cos(angle)])

    station_a_angle = 0
    station_a_position = position_from_angle(R_E, station_a_angle)
    station_central_angle = position_multiplier * length / R_E
    station_central_position = position_from_angle(R_E + ORBITAL_HEIGHT, station_central_angle)
    station_b_angle = length / R_E
    station_b_position = position_from_angle(R_E, station_b_angle)
    elevation_left = elevation_curved(ground_dist=np.abs(station_central_angle - station_a_angle) * R_E, h=ORBITAL_HEIGHT)
    elevation_right = elevation_curved(ground_dist=np.abs(station_b_angle - station_central_angle) * R_E, h=ORBITAL_HEIGHT)
    arrival_chance_left = (eta_dif(distance=distance(station_a_position, station_central_position),
                                   divergence_half_angle=DIVERGENCE_THETA,
                                   sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                   receiver_aperture_radius=RECEIVER_APERTURE_RADIUS,
                                   pointing_error_sigma=POINTING_ERROR_SIGMA)
                           * eta_atm(elevation=elevation_left))
    arrival_chance_right = (eta_dif(distance=distance(station_central_position, station_b_position),
                                    divergence_half_angle=DIVERGENCE_THETA,
                                    sender_aperture_radius=SENDER_APERTURE_RADIUS,
                                    receiver_aperture_radius=RECEIVER_APERTURE_RADIUS,
                                    pointing_error_sigma=POINTING_ERROR_SIGMA)
                            * eta_atm(elevation=elevation_right))

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1 - LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

    def time_distribution_left(source):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        comm_time = 2 * comm_distance / C
        eta = P_LINK * arrival_chance_left
        eta_effective = 1 - (1 - eta) * (1 - P_D)**2
        trial_time = T_P + comm_time  # I don't think that paper uses latency time and loading time?
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time, random_num

    def time_distribution_right(source):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        comm_time = 2 * comm_distance / C
        eta = P_LINK * arrival_chance_right
        eta_effective = 1 - (1 - eta) * (1 - P_D)**2
        trial_time = T_P + comm_time  # I don't think that paper uses latency time and loading time?
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time, random_num

    @lru_cache()  # CAREFUL: only makes sense if positions and errors do not change!
    def state_generation_left(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        storage_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:  # dephasing that has accrued while other qubit was travelling
                state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
            if station.dark_count_probability is not None:  # dark counts are handled here because the information about eta is needed for that
                eta = P_LINK * arrival_chance_left
                state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability))
        return state

    @lru_cache()  # CAREFUL: only makes sense if positions and errors do not change!
    def state_generation_right(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        storage_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:  # dephasing that has accrued while other qubit was travelling
                state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
            if station.dark_count_probability is not None:  # dark counts are handled here because the information about eta is needed for that
                eta = P_LINK * arrival_chance_right
                state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability))
        return state

    misalignment_noise = NoiseChannel(n_qubits=1, channel_function=construct_y_noise_channel(epsilon=E_MA))

    world = World()
    station_A = Station(world, position=station_a_position, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )
    station_central = Station(world, position=station_central_position,
                              memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP),
                              memory_cutoff_time=cutoff_time,
                              BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                              )
    station_B = Station(world, position=station_b_position, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )

    source_A = SchedulingSource(world, position=station_central_position,
                                target_stations=[station_A, station_central],
                                time_distribution=time_distribution_left,
                                state_generation=state_generation_left)
    source_B = SchedulingSource(world, position=station_central_position,
                                target_stations=[station_central, station_B],
                                time_distribution=time_distribution_right,
                                state_generation=state_generation_right)
    protocol = MultiMemoryProtocol(world, num_memories=num_memories)
    protocol.setup()

    # def step_check():
    #     protocol.check()
    #     world.print_status()
    #
    # def step_resolve():
    #     world.event_queue.resolve_next_event()
    #     world.print_status()
    #
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
    import matplotlib.pyplot as plt
    length_list = np.linspace(0, 4000e3, num=40)
    ps = [run(length=length, max_iter=1000, params={"P_LINK": 0.56, "T_DP": 1, "P_D": 10**-6, "ORBITAL_HEIGHT": 400e3, "SENDER_APERTURE_RADIUS": 0.15, "RECEIVER_APERTURE_RADIUS": 0.50, "DIVERGENCE_THETA": 10e-6}, cutoff_time=0.5, num_memories=1000) for length in length_list]
    from libs.aux_functions import standard_bipartite_evaluation
    res = [standard_bipartite_evaluation(p.data) for p in ps]
    plt.errorbar(length_list / 1000, [r[4] / 2 for r in res], yerr=[r[5] / 2 for r in res], fmt="o")  # 10 * np.log10(key_per_resource))
    plt.yscale("log")
    # plt.legend()
    plt.show()
