import unittest
from world import World
from events import EventQueue
from quantum_objects import Station, Source


class TestWorldSetup(unittest.TestCase):
    def setUp(self):
        self.world = World()

    def test_attributes(self):
        """Test for the existance of central attributes."""
        self.assertIsInstance(self.world.world_objects, dict)
        self.assertIsInstance(self.world.event_queue, EventQueue)

    def test_world_setup(self):
        """Test the creation of world objects via world methods."""
        # 1) Test stations.
        station = self.world.create_station(id=-1, position=0)
        self.assertIsInstance(station, Station)
        # test whether station is registered with the world
        station_objects = self.world.world_objects["Station"]
        self.assertIn(station, station_objects)
        # test deregistering when station is deleted
        station.destroy()
        self.assertEqual(station_objects, [])
        # create many stations
        num_stations = 20
        stations = [self.world.create_station(id=i, position=200 * i) for i in range(num_stations)]
        self.assertEqual(len(station_objects), num_stations)

        # 2) Test sources.
        source = self.world.create_source(position=300, target_stations=stations[0:2])
        self.assertIsInstance(source, Source)
        source_objects = self.world.world_objects["Source"]
        self.assertIn(source, source_objects)
        source.destroy()
        self.assertEqual(source_objects, [])


if __name__ == '__main__':
    unittest.main()
