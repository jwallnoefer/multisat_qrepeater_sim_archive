import sys
import abc
from abc import abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class AbstractEvent(ABC):
    def __init__(self, time, *args, **kwargs):
        self.time = time
        self.event_queue = None

    @abstractmethod
    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, *args, **kwargs)" % str(self.time)

    @abstractmethod
    def resolve(self):
        pass


class SourceEvent(AbstractEvent):
    def __init__(self, time, source, initial_state):
        self.source = source
        self.initial_state = initial_state
        super(SourceEvent, self).__init__(time)

    def __repr__(self):
        return self.__class__.__name__ + "(time=%s, source=%s, initial_state=%s)" % (str(self.time), str(self.source), repr(self.initial_state))

    def resolve(self):
        self.source.generate_pair(self.initial_state)


class EventQueue(object):
    def __init__(self):
        self.mylist = []
        self.current_time = 0

    def __str__(self):
        return "EventQueue: " + str(self.mylist)

    def add_event(self, event):
        event.event_queue = self
        self.mylist += [event]
        self.mylist.sort(key=lambda x: x.time)

    def resolve_next_event(self):
        event = self.mylist[0]
        event.resolve()
        self.current_time = event.time
        self.mylist = self.mylist[1:]
