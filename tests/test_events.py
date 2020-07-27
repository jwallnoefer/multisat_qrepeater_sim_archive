import unittest
from unittest.mock import MagicMock
from world import World
from events import Event, SourceEvent, EntanglementSwappingEvent, EventQueue, DiscardQubitEvent
from quantum_objects import Qubit, Pair
import numpy as np


class DummyEvent(Event):
    def __init__(self, time):
        super(DummyEvent, self).__init__(time)

    def __repr__(self):
        return ""

    def resolve(self):
        pass


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
