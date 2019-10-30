import sys
import abc
from abc import abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class WorldObject(ABC):
    def __init__(self, world):
        self.world = world
        self.world.register_world_object(self)

    def __del__(self):
        self.world.deregister_world_object(self)

    @property
    def event_queue(self):
        return self.world.event_queue


class Qubit(WorldObject):
    # not great yet, because creating a qubit this way, will not inform the station about it automatically
    def __init__(self, world, station):
        self.station = station
        super(Qubit, self).__init__(world)
        self.pair = None

    def __str__(self):
        return "Qubit at station %s, part of pair %s." % (str(self.station), str(self.pair))


class Pair(WorldObject):
    def __init__(self, world, qubits, initial_state):
        # maybe add a check that qubits are always in the same order?
        self.qubits = qubits
        self.state = initial_state
        self.qubit1.pair = self
        self.qubit2.pair = self
        super(Pair, self).__init__(world)

    # not sure we actually need to be able to change qubits
    @property
    def qubit1(self):
        return self.qubits[0]

    @qubit1.setter
    def qubit1(self, qubit):
        self.qubits[0] = qubit

    @property
    def qubit2(self):
        return self.qubits[1]

    @qubit2.setter
    def qubit2(self, qubit):
        self.qubits[1] = qubit


class Station(WorldObject):
    def __init__(self, world, id, position):
        self.id = id
        self.position = position
        # self.qubits = []
        super(Station, self).__init__(world)

    def __str__(self):
        return "Station with id %s at position %s." % (str(self.id), str(self.position))

    def create_qubit(self):
        new_qubit = Qubit(world=self.world, station=self)
        # self.qubits += [new_qubit]
        return new_qubit


class Source(WorldObject):
    def __init__(self, world, position, target_stations):
        self.position = position
        self.target_stations = target_stations
        super(Source, self).__init__(world)

    def generate_pair(self, initial_state):
        station1 = self.target_stations[0]
        station2 = self.target_stations[1]
        qubit1 = station1.create_qubit()
        qubit2 = station2.create_qubit()
        return Pair(world=self.world, qubits=[qubit1, qubit2], initial_state=initial_state)
