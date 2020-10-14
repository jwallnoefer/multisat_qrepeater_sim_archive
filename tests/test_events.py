import unittest
from unittest.mock import MagicMock
from world import World
from events import Event, SourceEvent, EntanglementSwappingEvent, EventQueue, DiscardQubitEvent, EntanglementPurificationEvent
from quantum_objects import Qubit, Pair, Station
import numpy as np
import libs.matrix as mat


class DummyEvent(Event):
    def __init__(self, time):
        super(DummyEvent, self).__init__(time)

    def __repr__(self):
        return ""

    def _main_effect(self):
        pass


def _known_dejmps_identical_copies(lambdas):
    # known formula for two identical states diagonal in the bell basis
    lambda_00, lambda_01, lambda_10, lambda_11 = lambdas
    new_lambdas = [lambda_00**2 + lambda_11**2, lambda_01**2 + lambda_10**2,
                   2 * lambda_00 * lambda_11, 2 * lambda_01 * lambda_10]
    p_suc = np.sum(new_lambdas)
    new_lambdas = np.array(new_lambdas) / p_suc
    return p_suc, new_lambdas

def _bogus_epp(rho):
    p_suc = 1
    state_after = np.dot(mat.phiplus, mat.H(mat.phiplus))
    return p_suc, state_after


class TestEvents(unittest.TestCase):
    # Not sure what else we could test here that does not boil down to asking
    # is the code exactly the code?
    def _aux_general_test(self, event):
        self.assertIsInstance(event, Event)

    def test_source_event(self):
        event = SourceEvent(time=0, source=MagicMock(), initial_state=MagicMock())
        self._aux_general_test(event)

    def test_entanglement_swapping_event(self):
        event = EntanglementSwappingEvent(time=0, pairs=MagicMock(), error_func=MagicMock())
        self._aux_general_test(event)

    def test_discard_qubit_event(self):
        world = World()
        qubit = Qubit(world=world, station=MagicMock())
        event = DiscardQubitEvent(time=0, qubit=qubit)
        self._aux_general_test(event)
        # now test whether qubit actually gets discarded
        self.assertIn(qubit, world.world_objects[qubit.type])
        event.resolve()
        self.assertNotIn(qubit, world.world_objects[qubit.type])
        # now test whether the whole pair gets discarded if a qubit is discarded
        qubits = [Qubit(world=world, station=MagicMock()) for i in range(2)]
        pair = Pair(world=world, qubits=qubits, initial_state=MagicMock())
        event = DiscardQubitEvent(time=0, qubit=qubits[0])
        self.assertIn(qubits[0], world.world_objects[qubits[0].type])
        self.assertIn(qubits[1], world.world_objects[qubits[1].type])
        self.assertIn(pair, world.world_objects[pair.type])
        event.resolve()
        self.assertNotIn(qubits[0], world.world_objects[qubits[0].type])
        self.assertNotIn(qubits[1], world.world_objects[qubits[1].type])
        self.assertNotIn(pair, world.world_objects[pair.type])


class TestEPP(unittest.TestCase):
    # entanglement purification gets own test case because there is more to test
    def setUp(self):
        self.world = World()
        self.station1 = Station(world=self.world, id=1, position=0)
        self.station2 = Station(world=self.world, id=2, position=100)

    # #  first we test the built_in epp modes
    def test_dejmps_epp_event(self):
        # test with the nice blackbox formula for two identical pairs
        for i in range(4):
            lambdas = np.random.random(4)
            lambdas[0] = lambdas[0] + 3
            lambdas = lambdas / np.sum(lambdas)
            initial_state = (lambdas[0] * np.dot(mat.phiplus, mat.H(mat.phiplus)) +
                             lambdas[1] * np.dot(mat.psiplus, mat.H(mat.psiplus)) +
                             lambdas[2] * np.dot(mat.phiminus, mat.H(mat.phiminus)) +
                             lambdas[3] * np.dot(mat.psiminus, mat.H(mat.psiminus))
                            )
            trusted_p_suc, trusted_lambdas = _known_dejmps_identical_copies(lambdas)
            trusted_state = (trusted_lambdas[0] * np.dot(mat.phiplus, mat.H(mat.phiplus)) +
                             trusted_lambdas[1] * np.dot(mat.psiplus, mat.H(mat.psiplus)) +
                             trusted_lambdas[2] * np.dot(mat.phiminus, mat.H(mat.phiminus)) +
                             trusted_lambdas[3] * np.dot(mat.psiminus, mat.H(mat.psiminus))
                            )
            self.world.world_objects["Pair"] = []
            while not self.world.world_objects["Pair"]: # do until successful, which should be soon for the way we did the coefficients
                pair1 = Pair(world=self.world, qubits=[self.station1.create_qubit(), self.station2.create_qubit()], initial_state=initial_state)
                pair2 = Pair(world=self.world, qubits=[self.station1.create_qubit(), self.station2.create_qubit()], initial_state=initial_state)
                event = EntanglementPurificationEvent(time=self.world.event_queue.current_time, pairs=[pair1, pair2], protocol="dejmps")
                self.world.event_queue.add_event(event)
                self.world.event_queue.resolve_next_event()  # resolve event
                self.world.event_queue.resolve_next_event()  # resole unblocking/discarding generated by the event
            self.assertEqual(len(self.world.world_objects["Pair"]), 1)
            pair = self.world.world_objects["Pair"][0]
            self.assertTrue(np.allclose(pair.state, trusted_state))
            # not sure how you would test success probability without making a huge number of cases

    # # test if the use of a bogus protocol works as expected
    def test_custom_epp_event(self):
        pair1 = Pair(world=self.world, qubits=[self.station1.create_qubit(), self.station2.create_qubit()], initial_state=np.dot(mat.phiminus, mat.H(mat.phiminus)))
        pair2 = Pair(world=self.world, qubits=[self.station1.create_qubit(), self.station2.create_qubit()], initial_state=np.dot(mat.psiplus, mat.H(mat.psiplus)))
        event = event = EntanglementPurificationEvent(time=self.world.event_queue.current_time, pairs=[pair1, pair2], protocol=_bogus_epp)
        self.world.event_queue.add_event(event)
        self.world.event_queue.resolve_next_event()  # resolve event
        self.world.event_queue.resolve_next_event()  # resole unblocking/discarding generated by the event
        self.assertEqual(len(self.world.world_objects["Pair"]), 1)
        pair = self.world.world_objects["Pair"][0]
        self.assertTrue(np.allclose(pair.state, np.dot(mat.phiplus, mat.H(mat.phiplus))))




class TestEventQueue(unittest.TestCase):
    def setUp(self):
        self.event_queue = EventQueue()

    def test_scheduling_events(self):
        dummy_event = DummyEvent(3.3)
        self.event_queue.add_event(dummy_event)
        self.assertIn(dummy_event, self.event_queue.queue)
        num_events = 30
        more_dummy_events = [DummyEvent(time=i) for i in range(num_events, 0, -1)]
        for event in more_dummy_events:
            self.event_queue.add_event(event)
        self.assertEqual(len(self.event_queue), num_events + 1)
        # trying to schedule event in the past
        with self.assertRaises(ValueError):
            self.event_queue.add_event(DummyEvent(time=-2))

    def test_resolving_events(self):
        mockEvent1 = MagicMock(time=0)
        mockEvent2 = MagicMock(time=1)
        self.event_queue.add_event(mockEvent2)
        self.event_queue.add_event(mockEvent1) # events added to queue in wrong order
        self.event_queue.resolve_next_event()
        mockEvent1.resolve.assert_called_once()
        mockEvent2.resolve.assert_not_called()
        self.event_queue.resolve_next_event()
        mockEvent2.resolve.assert_called_once()

    def test_resolve_until(self):
        num_events = 20
        mock_events = [MagicMock(time=i) for i in range(num_events)]
        for event in mock_events:
            self.event_queue.add_event(event)
        target_time=5
        self.event_queue.resolve_until(target_time)
        self.assertEqual(len(self.event_queue), num_events-(np.floor(target_time)+1))
        self.assertEqual(self.event_queue.current_time, target_time)
        with self.assertRaises(ValueError): # if given target_time in the past
            self.event_queue.resolve_until(0)


if __name__ == '__main__':
    unittest.main()
