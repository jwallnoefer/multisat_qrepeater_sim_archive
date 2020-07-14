import sys
import abc
from abc import abstractmethod
import libs.matrix as mat
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

    @abstractmethod
    def resolve(self):
        """Resolve the event.

        Returns
        -------
        None

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

    Attributes
    ----------
    source
    initial_state

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
        self.source.generate_pair(self.initial_state, *self.generation_args, **self.generation_kwargs)


class EntanglementSwappingEvent(Event):
    """Short summary.

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
        pass

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
        None

        """
        event = self.queue[0]
        self.current_time = event.time
        event.resolve()
        self.queue = self.queue[1:]

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
