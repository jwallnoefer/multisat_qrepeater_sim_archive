from events import EventQueue
from quantum_objects import Station, Source


class World(object):
    """A collection of WorldObjects with an EventQueue.

    The World can be understood as a central object describing an experimental
    setup. It keeps track of all WorldObjects, which contain all the information
    a protocol can access to make decisions.

    Attributes
    ----------
    event_queue : EventQueue
        A schedule of events so they can be resolved in order.
    world_objects : dict of str: WorldObject
        keys: str of WorldObject subclass names (obtained via x.__class__.__name__)
        values: list of WorldObjects

    """

    def __init__(self):
        self.event_queue = EventQueue()
        self.world_objects = {}  # should contain everything about the state world that the protocol needs to know

    def register_world_object(self, world_object):
        """Add a WorldObject to this world.

        Parameters
        ----------
        world_object : WorldObject
            WorldObject that should be added and tracked by `self.world_objects`

        Returns
        -------
        None

        """
        object_type = world_object.__class__.__name__
        if object_type not in self.world_objects:
            self.world_objects[object_type] = []
        self.world_objects[object_type] += [world_object]

    def deregister_world_object(self, world_object):
        """Remove a WorldObject from this World.

        Parameters
        ----------
        world_object : WorldObject
            The WorldObject that is removed from this World.

        Returns
        -------
        None

        """
        object_type = world_object.__class__.__name__
        type_list = self.world_objects[object_type]
        type_list.remove(world_object)

    def _create_world_object(self, ObjectClass, *args, **kwargs):
        return ObjectClass(self, *args, **kwargs)

    def create_station(self, id, position):
        """Create a Station object in this world.

        Parameters
        ----------
        id : int
            Numerical label for the station.
        position : scalar
            Position in meters in the 1D line for this linear repeater.

        Returns
        -------
        Station
            The newly created Station object.

        """
        return self._create_world_object(Station, id, position)

    def create_source(self, position, target_stations):
        """Create a Source of entangled pairs in this world.

        Parameters
        ----------
        position : scalar
            Position in meters in the 1D line for this linear repeater.
        target_stations : list of Stations
            The two stations the source to which the source sends the entangled
            pairs, usually the neighboring repeater stations.

        Returns
        -------
        Source
            The newly created Source object.

        """
        return self._create_world_object(Source, position, target_stations)
