import sys
import abc
from abc import abstractmethod
import libs.matrix as mat
from libs.aux_functions import dejmps_protocol
import numpy as np
import quantum_objects

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Event(ABC):
    """Abstract base class for events.

    Events are scheduled in an EventQueue and resolved at a specific time.

    Parameters
    ----------
    time : scalar
        The time at which the event will be resolved.

    Attributes
    ----------
    event_queue : EventQueue
        The event is part of this event queue.
        (None until added to an event queue.)
    time

    """

    def __init__(self, time, *args, **kwargs):
        self.time = time
        self.event_queue = None

    @abstractmethod
    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, *args, **kwargs)" % str(self.time)

    @property
    def type(self):
        """Returns the event type.

        Returns
        -------
        str
            The event type.

        """
        return self.__class__.__name__

    @abstractmethod
    def resolve(self):
        """Resolve the event.

        Returns
        -------
        None or dict
            dict may optionally be used to pass information to the protocol.
            The protocol will not necessarily use this information.

        """
        pass


class GenericEvent(Event):
    """Event that executes arbitrary function.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    resolve_function : callable
        Function that will be called when the resolve method is called.
    *args : any
        args for resolve_function.
    **kwargs : any
        kwargs for resolve_function.

    """
    def __init__(self, time, resolve_function, *args, **kwargs):
        self._resolve_function = resolve_function
        self._resolve_function_args = args
        self._resolve_function_kwargs = kwargs
        super(GenericEvent, self).__init__(time)

    def __repr__(self):
        return self.__class__.__name__ + "(time=" + str(time) + ", resolve_function="+str(resolve_function) + ", " + ", ".join(map(str, self._resolve_function_args)) + ", ".join(["%s=%s" % (str(k), str(v)) for k, v in self._resolve_function_kwargs.items()]) + ")"

    def resolve(self):
        return self._resolve_function(*self._resolve_function_args, **self._resolve_function_kwargs)


class SourceEvent(Event):
    """An Event generating an entangled pair.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    source : Source
        The source object generating the entangled pair.
    initial_state : np.ndarray
        Density matrix of the two qubit system being generated.
    *args, **kwargs :
        additional optional args and kwargs to pass to the the
        generate_pair method of `source`

    Attributes
    ----------
    source
    initial_state
    generation_args : additional args for the generate_pair method of source
    generation_kwargs : additional kwargs for the generate_pair method of source

    """

    def __init__(self, time, source, initial_state, *args, **kwargs):
        self.source = source
        self.initial_state = initial_state
        self.generation_args = args
        self.generation_kwargs = kwargs
        super(SourceEvent, self).__init__(time)

    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, source=%s, initial_state=%s)" % (str(self.time), str(self.source), repr(self.initial_state))

    def resolve(self):
        """Resolve the event.

        Generates a pair at the target stations of `self.source`.

        Returns
        -------
        None

        """
        # print("A source event happened at time", self.time, "while queue looked like this:", self.event_queue.queue)
        self.source.generate_pair(self.initial_state, *self.generation_args, **self.generation_kwargs)


class EntanglementSwappingEvent(Event):
    """An event to perform entanglement swapping.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    pairs : list of Pairs
        The left pair and the right pair.
    error_func : callable or None
        A four-qubit map. Default: None

    Attributes
    ----------
    pairs
    error_func

    """
    def __init__(self, time, pairs, error_func=None):
        self.pairs = pairs
        self.error_func = error_func  # currently a four-qubit channel, would be nicer as two-qubit channel that gets applied to the right qubits
        super(EntanglementSwappingEvent, self).__init__(time)

    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, pairs=%s, error_func=%s)" % (str(self.time), str(self.pairs), repr(self.error_func))

    def resolve(self):
        """Resolve the event.

        Performs entanglement swapping between the two pairs and generates the
        appropriate Pair object for the long-distance pair.

        Returns
        -------
        None

        """
        # it would be nice if this could handle arbitrary configurations
        # instead of relying on strict indexes of left and right pairs
        left_pair = self.pairs[0]
        right_pair = self.pairs[1]
        assert left_pair.qubits[1].station is right_pair.qubits[0].station
        left_pair.update_time()
        right_pair.update_time()
        four_qubit_state = mat.tensor(left_pair.state, right_pair.state)
        # non-ideal-bell-measurement
        if self.error_func is not None:
            four_qubit_state = self.error_func(four_qubit_state)
        my_proj = mat.tensor(mat.I(2), mat.phiplus, mat.I(2))
        two_qubit_state = np.dot(np.dot(mat.H(my_proj), four_qubit_state), my_proj)
        two_qubit_state = two_qubit_state / np.trace(two_qubit_state)
        new_pair = quantum_objects.Pair(world=left_pair.world, qubits=[left_pair.qubits[0], right_pair.qubits[1]],
                                        initial_state=two_qubit_state,
                                        initial_cost_add=left_pair.resource_cost_add + right_pair.resource_cost_add,
                                        initial_cost_max=max(left_pair.resource_cost_max, right_pair.resource_cost_max))
        # cleanup
        left_pair.qubits[1].destroy()
        right_pair.qubits[0].destroy()
        left_pair.destroy()
        right_pair.destroy()


class DiscardQubitEvent(Event):
    """Event to discard a qubit and associated pair.

    For example if the qubit sat in memory too long and is discarded.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    qubit : Qubit
        The Qubit that will be discarded.

    Attributes
    ----------
    qubit

    """
    def __init__(self, time, qubit):
        self.qubit = qubit
        super(DiscardQubitEvent, self).__init__(time)

    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, qubit=%s)" % (str(self.time), str(self.qubit))

    def resolve(self):
        """Discards the qubit and associated pair, if the qubit still exists.

        Returns
        -------
        None

        """
        if self.qubit in self.qubit.world.world_objects["Qubit"]:  # only do something if qubit still exists
            if self.qubit.pair is not None:
                self.qubit.pair.destroy_and_track_resources()
                self.qubit.pair.qubits[0].destroy()
                self.qubit.pair.qubits[1].destroy()
            else:
                self.qubit.destroy()
            # print("A Discard Event happened with eventqueue:", self.qubit.world.event_queue.queue)


class EntanglementPurificationEvent(Event):
    """Short summary.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    pairs : list of Pairs
        The pairs involved in the entanglement purification protocol.
    protocol : {"dejmps"} or callable
        Can be one of the pre-installed or an arbitrary callable that takes
        a tensor product of pair states as input and returns a tuple of
        (success probability, state of a single pair) back.
        So far only supports n->1 protocols.


    Attributes
    ----------
    pairs
    protocol

    """
    def __init__(self, time, pairs, protocol="dejmps"):
        self.pairs = pairs
        if protocol == "dejmps":
            self.protocol = dejmps_protocol
        elif callable(protocol):
            self.protocol = protocol
        else:
            raise ValueError("EntanglementPurificationEvent got a protocol type that is not supported: " + repr(protocol))
        super(EntanglementPurificationEvent, self).__init__(time)

    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, pairs=%s, protocol=%s)" % (repr(self.time), repr(self.pairs), repr(self.protocol))

    def resolve(self):
        """Probabilistically performs the entanglement purification protocol.

        Returns
        -------
        None

        """
        # probably could use a check that pairs are between same stations?
        for pair in self.pairs:
            pair.update_time()
        rho = mat.tensor(*[pair.state for pair in self.pairs])
        p_suc, state = self.protocol(rho)
        if np.random.random() <= p_suc:  # if successful
            output_pair = self.pairs[0]
            output_pair.state = state
            if output_pair.resource_cost_add is not None:
                output_pair.resource_cost_add = np.sum([pair.resource_cost_add for pair in self.pairs])
            if output_pair.resource_cost_max is not None:
                output_pair.resource_cost_max = np.sum([pair.resource_cost_max for pair in self.pairs])
            for pair in self.pairs[1:]:  # pairs that have been destroyed in the process
                pair.qubits[0].destroy()
                pair.qubits[1].destroy()
                pair.destroy()
            return {"event_type": self.type, "output_pair": output_pair, "is_successful": True}
        else:  # if unsuccessful
            for pair in self.pairs:  # destroy all the involved pairs but track resources
                pair.destroy_and_track_resources()
                pair.qubits[0].destroy()
                pair.qubits[1].destroy()
            return {"event_type": self.type, "output_pair": None, "is_successful": False}

class EventQueue(object):
    """Provides methods to queue and resolve Events in order.

    Attributes
    ----------
    queue : list of Events
        An ordered list of future events to resolve.
    current_time : scalar
        The current time of the event queue.

    """

    def __init__(self):
        self.queue = []
        self.current_time = 0

    def __str__(self):
        return "EventQueue: " + str(self.queue)

    def __len__(self):
        return len(self.queue)

    @property
    def next_event(self):
        """Helper property to access next scheduled event.

        Returns
        -------
        Event or None
            The next scheduled event. None if the event queue is empty.

        """
        try:
            return self.queue[0]
        except IndexError:
            return None

    def add_event(self, event):
        """Add an event to the queue.

        The queue is sorted again in order to schedule the event at the correct
        time.

        Parameters
        ----------
        event : Event
            The Event to be added to the queue.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `event.time` is in the past.

        """
        if event.time < self.current_time:
            raise ValueError("EventQueue.add_event tried to schedule an event in the past.")
        event.event_queue = self
        self.queue += [event]
        self.queue.sort(key=lambda x: x.time)

    def resolve_next_event(self):
        """Remove the next scheduled event from the queue and resolve it.

        Returns
        -------
        None or dict:
            Whatever event.resolve() returns (usually None or dict). Is used to pass resolve message
            through to the protocol.

        """
        event = self.queue[0]
        if isinstance(event, DiscardQubitEvent):
            # try to find another type of event at the same time
            try:
                better_event = next(filter(lambda x: x.time == event.time and not isinstance(x, DiscardQubitEvent), self.queue))
                self.queue.remove(better_event)
                self.queue.insert(0, better_event)
                event = self.queue[0]
            except StopIteration:
                pass
        self.current_time = event.time
        return_message = event.resolve()
        self.queue = self.queue[1:]
        return return_message

    def resolve_until(self, target_time):
        """Resolve events until `target_time` is reached.

        Parameters
        ----------
        target_time : scalar
            Resolve until current_time is this.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `target_time` lies in the past.
        """
        if target_time < self.current_time:
            raise ValueError("EventQueue.resolve_until cannot resolve to a time in the past.")
        while self.queue:
            event = self.queue[0]
            if event.time <= target_time:
                self.resolve_next_event()
            else:
                break
        self.current_time = target_time

    def advance_time(self, time_interval):
        """Helper method to manually advance time.

        Parameters
        ----------
        time_interval : int
            The amount of time that passes.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an event is skipped during the `time_interval`.
        """
        self.current_time += time_interval
        if self.queue and self.queue[0].time < self.current_time:
            raise ValueError("time_interval too large. Manual time advancing skipped an event. Time travel is not permitted.")
