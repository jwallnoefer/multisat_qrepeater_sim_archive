from events import EventQueue

import sys
import abc

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class World(object):
    def __init__(self):
        self.event_queue = EventQueue()
        self.world_objects = {}  # should contain everything about the state world that the protocol needs to know

    def register_world_object(self, world_object):
        pass

    def deregister_world_object(self, world_object):
        pass


class WorldObject(ABC):
    def __init__(self, world):
        self.world = world
        self.world.register_world_object(self)

    def __del__(self):
        self.world.deregister_world_object(self)

    @property
    def event_queue(self):
        return self.world.event_queue
