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


if __name__ == '__main__':
    unittest.main()
