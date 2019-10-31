import unittest
from unittest.mock import MagicMock
from world import World
from events import Event, SourceEvent, EventQueue


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

    def test_resolving_events(self):
        mockEvent1 = MagicMock(time=0)
        mockEvent2 = MagicMock(time=1)
        self.event_queue.add_event(mockEvent2)
        self.event_queue.add_event(mockEvent1)
        self.event_queue.resolve_next_event()
        mockEvent1.resolve.assert_called_once()
        mockEvent2.resolve.assert_not_called()
        self.event_queue.resolve_next_event()
        mockEvent2.resolve.assert_called_once()


if __name__ == '__main__':
    unittest.main()
