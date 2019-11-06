import unittest
from unittest.mock import MagicMock
from world import World
from quantum_objects import Qubit, Pair, Station, Source, WorldObject
import numpy as np


class TestQuantumObjects(unittest.TestCase):
    def setUp(self):
        self.world = World()

    def _aux_general_test(self, quantum_object):
        self.assertIsInstance(quantum_object, WorldObject)
        # assert that quantum object is registered
        self.assertIn(quantum_object, self.world.world_objects[quantum_object.__class__.__name__])
        # assert that object is in test world and has correct evenetqueue
        self.assertIs(quantum_object.world, self.world)
        self.assertIs(quantum_object.event_queue, self.world.event_queue)

    def test_qubit(self):
        qubit = Qubit(world=self.world, station=MagicMock())
        self._aux_general_test(qubit)

    def test_pair(self):
        qubits = [Qubit(world=self.world, station=MagicMock()) for i in range(2)]
        pair = Pair(world=self.world, qubits=qubits, initial_state=np.diag([1, 0, 0, 0]))
        self._aux_general_test(pair)

    def test_station(self):
        station = Station(world=self.world, id=1, position=0)
        self._aux_general_test(station)
        qubit = station.create_qubit()
        self.assertIsInstance(qubit, Qubit)
        self.assertIs(qubit.station, station)

    def test_source(self):
        stations = [Station(world=self.world, id=i, position=200 * i) for i in range(2)]
        source = Source(world=self.world, position=100, target_stations=stations)
        self._aux_general_test(source)
        test_state = np.random.rand(4, 4)
        pair = source.generate_pair(test_state)
        self.assertIsInstance(pair, Pair)
        self.assertTrue(np.allclose(pair.state, test_state))
        pair_stations = [qubit.station for qubit in pair.qubits]
        self.assertEqual(pair_stations, stations)


if __name__ == '__main__':
    unittest.main()
