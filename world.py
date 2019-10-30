from events import EventQueue
from quantum_objects import Station, Source


class World(object):
    def __init__(self):
        self.event_queue = EventQueue()
        self.world_objects = {}  # should contain everything about the state world that the protocol needs to know

    def register_world_object(self, world_object):
        object_type = world_object.__class__.__name__
        if object_type not in self.world_objects:
            self.world_objects[object_type] = []
        self.world_objects[object_type] += [world_object]

    def deregister_world_object(self, world_object):
        object_type = world_object.__class__.__name__
        type_list = self.world_objects[object_type]
        type_list.remove(world_object)

    def _create_world_object(self, ObjectClass, *args, **kwargs):
        return ObjectClass(self, *args, **kwargs)

    def create_station(self, id, position):
        return self._create_world_object(Station, id, position)

    def create_source(self, position, target_stations):
        return self._create_world_object(Source, position, target_stations)
