import sys
import abc

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class AbstractEvent(ABC):
    def __init__(self, time, event_queue, *args, **kwargs):
        self.time = time
        self.event_queue = event_queue

    @abc.abstractmethod
    def resolve(self):
        pass


class SourceEvent(AbstractEvent):
    def __init__(self, time, event_queue, *args, **kwargs):
        # do something with args and kwargs
        AbstractEvent.__init__(self, time, event_queue)

    def resolve(self):
        pass


class EventQueue(object):
    def __init__(self):
        self.mylist = []
        self.current_time = 0

    def add_event(self, event):
        self.mylist += [event]
        self.mylist.sort(key=lambda x: x.start_time)

    def resolve_next_event(self):
        event = self.mylist[0]
        event.event_function(event.involved_qubits)
        self.mylist = self.mylist[1:]
