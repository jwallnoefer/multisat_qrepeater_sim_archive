import sys
import abc
from abc import abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Event(ABC):
    """Abstract base class for events.

    Events are scheduled in an EventQueue and resolved at a specific time.
    """

    def __init__(self, time, *args, **kwargs):
        self.time = time
        self.event_queue = None

    @abstractmethod
    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, *args, **kwargs)" % str(self.time)

    @abstractmethod
    def resolve(self):
        """Resolve the event."""
        pass


class GenericEvent(Event):
    """Event that executes arbitrary function.

    Args:
        time (scalar): Time at which the event will be resolved.
        resolve_function (callable): Function that will be called when the resolve method is called.
        *args: args for resolve_function
        **kwargs: kwargs for resolve_function

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

    def __init__(self, time, source, initial_state):
        self.source = source
        self.initial_state = initial_state
        super(SourceEvent, self).__init__(time)

    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, source=%s, initial_state=%s)" % (str(self.time), str(self.source), repr(self.initial_state))

    def resolve(self):
        """Resolve the event.

        Generates a pair at the target stations of `self.source`
        """
        self.source.generate_pair(self.initial_state)


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

        """
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
        event.resolve()
        self.current_time = event.time
        self.queue = self.queue[1:]
