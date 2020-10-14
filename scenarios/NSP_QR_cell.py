import os, sys; sys.path.insert(0, os.path.abspath("."))
from quantum_objects import Source, SchedulingSource, Station, Pair
from protocol import Protocol
from world import World
from events import SourceEvent, GenericEvent, EntanglementSwappingEvent
import libs.matrix as mat
import numpy as np
from libs.aux_functions import apply_single_qubit_map, x_noise_channel, y_noise_channel, z_noise_channel, w_noise_channel, assert_dir
import matplotlib.pyplot as plt
from warnings import warn

# result_path = os.path.join("results", "luetkenhaus")
#
# ETA_P = 0.66  # preparation efficiency
# T_P = 2 * 10**-6  # preparation time
# ETA_C = 0.04 * 0.3  # phton-fiber coupling efficiency * wavelength conversion
# T_2 = 1  # dephasing time
C = 2 * 10**8 # speed of light in optical fiber
L_ATT = 22 * 10**3  # attenuation length
# E_M_A = 0.01  # misalignment error
# P_D = 10**-8  # dark count probability per detector
# ETA_D = 0.3  # detector efficiency
# P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
# LAMBDA_BSM = 0.97  # BSM ideality parameter
# F = 1.16  # error correction inefficiency
#
# ETA_TOT = ETA_P * ETA_C * ETA_D


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t/dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return dephasing_noise_channel

def run(length, max_iter, params, cutoff_time=None, mode="sim"):
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

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1-LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

    def alpha_of_eta(eta):
        return eta * (1 - P_D) / (1 - (1 - eta) * (1 - P_D)**2)

    def time_distribution(source):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        comm_time = 2 * comm_distance / C
        eta = P_LINK * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - P_D)**2
        trial_time = T_P + comm_time  # I don't think that paper uses latency time and loading time?
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time, random_num

    def state_generation(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        # TODO needs more sophisticated handling for other scenarios - especially if not only the central station is faulty
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        storage_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:  # only central station has noisy storage
                state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
            if station.memory_noise is None:  # only count misalignment and dark counts for end stations
                # misalignment
                state = apply_single_qubit_map(map_func=y_noise_channel, qubit_index=idx, rho=state, epsilon=E_MA)
                eta = P_LINK * np.exp(-comm_distance / L_ATT)
                # dark counts are modeled as white noise
                state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta))
        return state

    class TwoLinkProtocol(Protocol):
        """Short summary.

        Parameters
        ----------
        world : World
            The world in which the protocol will be performed.
        mode : {"seq", "sim"}
            Selects sequential or simultaneous generation of links.

        Attributes
        ----------
        time_list : list of scalars
        fidelity_list : list of scalars
        correlations_z_list : list of scalars
        correlations_x_list : list of scalars
        resource_cost_max_list : list of scalars
        mode : str
            "seq" or "sim"

        """
        def __init__(self, world, mode="seq"):
            if mode != "seq" and mode != "sim":
                raise ValueError("LuetkenhausProtocol does not support mode %s. Use \"seq\" for sequential state generation, or \"sim\" for simultaneous state generation.")
            self.mode = mode
            self.time_list = []
            self.fidelity_list = []
            self.correlations_z_list = []
            self.correlations_x_list = []
            self.resource_cost_max_list = []
            super(TwoLinkProtocol, self).__init__(world)

        def setup(self):
            """Identifies the stations and sources in the world.

            Should be run after the relevant WorldObjects have been added
            to the world.

            Returns
            -------
            None

            """
            stations = self.world.world_objects["Station"]
            assert len(stations) == 3
            self.station_A, self.station_central, self.station_B = sorted(stations, key=lambda x: x.position)
            sources = self.world.world_objects["Source"]
            assert len(sources) == 2
            self.source_A = next(filter(lambda source: self.station_A in source.target_stations and self.station_central in source.target_stations, sources))
            self.source_B = next(filter(lambda source: self.station_central in source.target_stations and self.station_B in source.target_stations, sources))
            assert callable(getattr(self.source_A, "schedule_event", None))  # schedule_event is a required method for this protocol
            assert callable(getattr(self.source_B, "schedule_event", None))


        def _pair_is_between_stations(self, pair, station1, station2):
            return (pair.qubit1.station == station1 and pair.qubit2.station == station2) or (pair.qubit1.station == station2 and pair.qubit2.station == station1)

        def _get_left_pair(self):
            try:
                pairs = self.world.world_objects["Pair"]
            except KeyError:
                pairs = []
            try:
                return next(filter(lambda x: self._pair_is_between_stations(x, self.station_A, self.station_central), pairs))
            except StopIteration:
                return None

        def _get_right_pair(self):
            try:
                pairs = self.world.world_objects["Pair"]
            except KeyError:
                pairs = []
            try:
                return next(filter(lambda x: self._pair_is_between_stations(x, self.station_central, self.station_B), pairs))
            except StopIteration:
                return None

        def _get_long_range_pair(self):
            try:
                pairs = self.world.world_objects["Pair"]
            except KeyError:
                pairs = []
            try:
                return next(filter(lambda x: self._pair_is_between_stations(x, self.station_A, self.station_B), pairs))
            except StopIteration:
                return None

        def _left_pair_is_scheduled(self):
            try:
                next(filter(lambda event: isinstance(event, SourceEvent) and
                                          (self.station_A in event.source.target_stations) and
                                          (self.station_central in event.source.target_stations),
                            self.world.event_queue.queue))
                return True
            except StopIteration:
                return False

        def _right_pair_is_scheduled(self):
            try:
                next(filter(lambda event: isinstance(event, SourceEvent) and
                                          (self.station_central in event.source.target_stations) and
                                          (self.station_B in event.source.target_stations),
                            self.world.event_queue.queue))
                return True
            except StopIteration:
                return False


        def _eval_pair(self, long_range_pair):
            comm_distance = np.max([np.abs(self.station_central.position - self.station_A.position), np.abs(self.station_B.position - self.station_central.position)])
            comm_time = comm_distance / C

            pair_fidelity = np.real_if_close(np.dot(np.dot(mat.H(mat.phiplus), long_range_pair.state), mat.phiplus))[0, 0]
            self.time_list += [self.world.event_queue.current_time + comm_time]
            self.fidelity_list += [pair_fidelity]

            z0z0 = mat.tensor(mat.z0, mat.z0)
            z1z1 = mat.tensor(mat.z1, mat.z1)
            correlations_z = np.real_if_close(np.dot(np.dot(mat.H(z0z0), long_range_pair.state), z0z0)[0, 0] +  np.dot(np.dot(mat.H(z1z1), long_range_pair.state), z1z1))[0, 0]
            self.correlations_z_list += [correlations_z]

            x0x0 = mat.tensor(mat.x0, mat.x0)
            x1x1 = mat.tensor(mat.x1, mat.x1)
            correlations_x = np.real_if_close(np.dot(np.dot(mat.H(x0x0), long_range_pair.state), x0x0)[0, 0] +  np.dot(np.dot(mat.H(x1x1), long_range_pair.state), x1x1))[0, 0]
            self.correlations_x_list += [correlations_x]

            self.resource_cost_max_list += [long_range_pair.resource_cost_max]
            return

        def check(self):
            """Checks world state and schedules new events.

            Summary of the Protocol:
            Establish a left link and a right link.
            Then perform entanglement swapping.
            Record metrics about the long distance pair.
            Repeat.
            However, sometimes pairs will be discarded because of memory cutoff
            times, which means we need to check regularly if that happened.

            Returns
            -------
            None

            """
            try:
                pairs = self.world.world_objects["Pair"]
            except KeyError:
                pairs = []
            left_pair = self._get_left_pair()
            right_pair = self._get_right_pair()
            # schedule pair creation if there is no pair and pair creation is not already scheduled
            if self.mode == "sim":
                if not left_pair and not self._left_pair_is_scheduled():
                    self.source_A.schedule_event()
                if not right_pair and not self._right_pair_is_scheduled():
                    self.source_B.schedule_event()
            elif self.mode == "seq":
                if not pairs and not self._left_pair_is_scheduled() and not self._right_pair_is_scheduled():
                    self.source_A.schedule_event()
                elif left_pair and not right_pair and not self._right_pair_is_scheduled():
                    self.source_B.schedule_event()
                elif right_pair and not left_pair and not self._left_pair_is_scheduled():  # this might happen if left pair is discarded
                    self.source_A.schedule_event()
            # rest continues the same for both modes
            if right_pair and left_pair:
                ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time, pairs=[left_pair, right_pair], error_func=imperfect_bsm_err_func)
                # print("an entswap event was scheduled at %.8f while event_queue looked like this:" % self.world.event_queue.current_time, self.world.event_queue.queue)
                self.world.event_queue.add_event(ent_swap_event)
                return
            long_range_pair = self._get_long_range_pair()
            if long_range_pair:
                self._eval_pair(long_range_pair)
                # cleanup
                long_range_pair.qubits[0].destroy()
                long_range_pair.qubits[1].destroy()
                long_range_pair.destroy()
                self.check()
                return
            if not self.world.event_queue.queue:
                warn("Protocol may be stuck in a state without events.")

    world = World()
    station_A = Station(world, id=0, position=0, memory_noise=None)
    station_B = Station(world, id=1, position=length, memory_noise=None)
    station_central = Station(world, id=2, position=length/2, memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP), memory_cutoff_time=cutoff_time)
    source_A = SchedulingSource(world, position=length/2, target_stations=[station_A, station_central], time_distribution=time_distribution, state_generation=state_generation)
    source_B = SchedulingSource(world, position=length/2, target_stations=[station_central, station_B], time_distribution=time_distribution, state_generation=state_generation)
    protocol = TwoLinkProtocol(world, mode=mode)
    protocol.setup()

    while len(protocol.time_list) < max_iter:
        protocol.check()
        world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":
    run(length=22000, max_iter=100, params={}, mode="sim")
