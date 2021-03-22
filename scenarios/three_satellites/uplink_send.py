import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from world import World
from protocol import Protocol
from events import Event, DiscardQubitEvent, EntanglementSwappingEvent, SourceEvent
from copy import copy
from libs.aux_functions import distance
from consts import SPEED_OF_LIGHT_IN_VACCUM as C


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
        self.event_queue.add_event(new_event)
        return return_value


class SendOnScheduleEvent(RepeatingEvent):
    def __init__(self, time, offset_time, station, target_station, channel_eta_func, priority=21, ignore_blocked=False):
        self.station = station
        self.target_station = target_station
        self.channel_eta_func = channel_eta_func
        super(RepeatingEvent, self).__init__(time=time, offset_time=offset_time, required_objects=[station], priority=priority, ignore_blocked=ignore_blocked)

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, offset_time={self.offset_time}, station={self.station}, target_station={self.target_station}, channel_eta_func={self.channel_eta_func}, priority={self.priority}, ignore_blocked={self.ignore_blocked})"

    def _main_effect(self):
        for qubit in self.station.qubits:
            send_distance = distance(qubit.station, self.station)
            arrive_time = self.world.event_queue.current_time + send_distance / C
            p_suc = self.channel_eta_func(send_distance)
            self.station.remove_qubit(qubit)
            qubit.station = None
            qubit.is_blocked = True
            if np.random.random() <= p_suc:
                event = QubitArrivesAtStationEvent(time=arrive_time, qubit=qubit, station=self.target_station)
                self.world.event_queue.add_event(event)
            else:
                event = DiscardQubitEvent(time=arrive_time, qubit=qubit)
                self.world.event_queue.add_event(event)
        return {"event_type": self.type, "resolve_successful": True}


class QubitArrivesAtStationEvent(Event):
    def __init__(self, time, qubit, station, priority=20, ignore_blocked=True):
        self.qubit = qubit
        self.station = station
        super(QubitArrivesAtStationEvent, self).__init__(time=time, required_objects=[qubit, station], priority=priority, ignore_blocked=ignore_blocked)

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, qubit={self.qubit}, station={self.station}, priority={self.priority}, ignore_blocked={self.ignore_blocked})"

    def _main_effect(self):
        self.station.qubits += [self.qubit]
        self.qubit.station = self.station
        self.qubit.is_blocked = False
        if self.station.memory_cutoff_time is not None:
            discard_event = DiscardQubitEvent(time=self.event_queue.current_time + self.station.memory_cutoff_time, qubit=self.qubit)
            self.event_queue.add_event(discard_event)
        return {"event_type": self.type, "resolve_successful": True}


class UplinkSendProtocol(Protocol):
    def __init__(self, world, stations, sources, send_time, channel_eta_func):
        self.time_list = []
        self.state_list = []
        self.stations = stations
        self.sources = sources
        self.send_time = send_time
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


def run():
    # unpack params
    # do setup
    # start the repeating events off
    # go



if __name__ == "__main__":
    pass
