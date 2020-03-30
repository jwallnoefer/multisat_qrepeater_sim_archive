import sys
import abc
from abc import abstractmethod
from warnings import warn
from aux_functions import apply_single_qubit_map

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class WorldObject(ABC):
    """Abstract base class for objects that exist within a World.

    This ensures that all WorldObjects are known by the associated World and
    that they have easy access via properties to the world and its associated
    event_queue.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.

    Attributes
    ----------
    world : World
    event_queue : EventQueue
    last_updated : scalar

    """

    def __init__(self, world):
        self.world = world
        self.world.register_world_object(self)
        self.last_updated = self.event_queue.current_time

    def destroy(self):
        """Remove this WordlObject from the world."""
        # in the future it might be nice to also remove associated events etc.
        self.world.deregister_world_object(self)

    @property
    def event_queue(self):
        """Shortcut to access the event_queue `self.world.event_queue`.

        Returns
        -------
        EventQueue
            The event queue

        """
        return self.world.event_queue

    def _on_update_time(self):
        pass

    def update_time(self):  # to be used to update internal time
        self._on_update_time()
        self.last_updated = self.event_queue.current_time


class Qubit(WorldObject):
    """A Qubit.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    station : Station
        The station at which the qubit is located.

    Attributes
    ----------
    pair : Pair or None
        Pair if the qubit is part of a Pair, None else.
    station : Station
        The station at which the qubit is located.

    """

    # station should also know about which qubits are at its location
    def __init__(self, world, station):
        self.station = station
        super(Qubit, self).__init__(world)
        self.pair = None

    def __str__(self):
        return "Qubit at station %s, part of pair %s." % (str(self.station), str(self.pair))

    def destroy(self):
        # station needs to be notified that qubit is no longer there, not sure how to handle pairs yet
        self.station.remove_qubit(self)
        super(Qubit, self).destroy()


class Pair(WorldObject):
    """A Pair of two qubits with its associated quantum state.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    qubits : list of Qubits
        The two qubits that are part of this entangled Pair.
    initial_state : np.ndarray
        The two qubit system is intialized with this density matrix.

    Attributes
    ----------
    state : np.ndarray
        Current density matrix of this two qubit system.
    qubit1 : Qubit
        Alternative way to access `self.qubits[0]`
    qubit2 : Qubit
        Alternative way to access `self.qubits[1]`
    qubits : List of qubits
        The two qubits that are part of this entangled Pair.

    """

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
        """Alternative way to access `self.qubits[0]`.

        Returns
        -------
        Qubit
            The first qubit of the pair.

        """
        return self.qubits[0]

    @qubit1.setter
    def qubit1(self, qubit):
        self.qubits[0] = qubit

    @property
    def qubit2(self):
        """Alternative way to access `self.qubits[1]`.

        Returns
        -------
        Qubit
            The second qubit of the pair.

        """
        return self.qubits[1]

    @qubit2.setter
    def qubit2(self, qubit):
        self.qubits[1] = qubit

    def _on_update_time(self):
        time_interval = self.event_queue.current_time - self.last_updated
        map0 = self.qubits[0]
        if map0 is not None:
            self.state = apply_single_qubit_map(map_func=map0, qubit_index=0, rho=self.state, t=time_interval)
        map1 = self.qubits[1]
        if map1 is not None:
            self.state = apply_single_qubit_map(map_func=map1, qubit_index=1, rho=self.state, t=time_interval)


class Station(WorldObject):
    """A repeater station.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    id : int0
        Numerical label for the station.
    position : scalar
        Position in meters in the 1D line for this linear repeater.

    Attributes
    ----------
    id : int
        Numerical label for the station.
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    qubits : list of Qubit objects
        The qubits currently at this position.

    """

    def __init__(self, world, id, position, memory_noise=None):
        self.id = id
        self.position = position
        self.qubits = []
        self.memory_noise = memory_noise
        super(Station, self).__init__(world)

    def __str__(self):
        return "Station with id %s at position %s." % (str(self.id), str(self.position))

    def create_qubit(self):
        """Create a new qubit at this station.

        Returns
        -------
        Qubit
            The created Qubit object.

        """
        new_qubit = Qubit(world=self.world, station=self)
        self.qubits += [new_qubit]
        return new_qubit

    def remove_qubit(self, qubit):
        try:
            self.qubits.remove(qubit)
        except ValueError:
            warn("Tried to remove qubit %s from station %s, but the station was not tracking that qubit." % (repr(qubit), repr(self)))


class Source(WorldObject):
    """A source of entangled pairs.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The two stations the source to which the source sends the entangled
        pairs, usually the neighboring repeater stations.

    Attributes
    ----------
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The two stations the source to which the source sends the entangled
        pairs, usually the neighboring repeater stations.

    """

    def __init__(self, world, position, target_stations):
        self.position = position
        self.target_stations = target_stations
        super(Source, self).__init__(world)

    def generate_pair(self, initial_state):
        """Generate an entangled pair.

        The Pair will be generated in the `initial_state` at the
        `self.target_stations` of the source.
        Usually called from a SourceEvent.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial density matrix of the two-qubit

        Returns
        -------
        Pair
            The newly generated Pair.

        """
        station1 = self.target_stations[0]
        station2 = self.target_stations[1]
        qubit1 = station1.create_qubit()
        qubit2 = station2.create_qubit()
        return Pair(world=self.world, qubits=[qubit1, qubit2], initial_state=initial_state)
