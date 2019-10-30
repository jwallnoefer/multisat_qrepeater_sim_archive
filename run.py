from quantum_objects import Pair, Qubit, Station, Source
from events import SourceEvent, EventQueue
from world import World
import numpy as np


def tensor(*args):
    '''
    Returns the matrix representation of the tensor product of an arbitrary
    number of matrices.
    '''
    res = np.array([[1]])
    for i in args:
        res = np.tensordot(res, i, ([], [])).swapaxes(1, 2).reshape(res.shape[0] * i.shape[0], res.shape[1] * i.shape[1])
    return res


def H(rho):  # Hermitian Conjugate - because only matrices are allowed to matrix.H
    return rho.conj().T


z0 = np.array([1, 0]).reshape(2, 1)
z1 = np.array([0, 1]).reshape(2, 1)

phiplus = 1 / np.sqrt(2) * (tensor(z0, z0) + tensor(z1, z1))


if __name__ == "__main__":
    world = World()
    station1 = world.create_station(id=1, position=0)
    station2 = world.create_station(id=2, position=2000)  # 2 km distance
    my_source = world.create_source(position=1000, target_stations=[station1, station2])  # put a source in the middle
    initial_state = np.dot(phiplus, H(phiplus))
    # core loop
    while world.event_queue.current_time < 600:  # we give it ten minutes
        new_pair_event = SourceEvent(time=world.event_queue.current_time + 30, source=my_source, initial_state=initial_state)  # continously generating every 30 seconds
        world.event_queue.add_event(new_pair_event)
        print("The time is: " + str(world.event_queue.current_time))
        print(world.event_queue)
        print("World objects: " + str(world.world_objects))
        world.event_queue.resolve_next_event()
        input("Press Enter to continue...")
