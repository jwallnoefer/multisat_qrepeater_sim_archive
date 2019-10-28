from world_object import WorldObject


class Qubit(WorldObject):
    # not great yet, because creating a qubit this way, will not inform the station about it automatically
    def __init__(self, world, station):
        self.station = station
        super(Qubit, self).__init__(self, world)


class Pair(WorldObject):
    def __init__(self, world, qubits, initial_state):
        # maybe add a check that qubits are always in the same order?
        self.qubits = qubits
        self.state = initial_state
        super(Pair, self).__init__(self, world)

    @property
    def qubit1(self):
        return self.qubits[0]

    @qubit1.setter
    def qubit1(self, qubit):
        self.qubits[0] = qubit

    @property
    def qubit2(self):
        return self.qubits[1]

    @qubit2.setter
    def qubit2(self, qubit):
        self.qubits[1] = qubit


class Station(WorldObject):
    def __init__(self, world, id, position):
        self.id = id
        self.position = position
        self.qubits = []
        super(Station, self).__init__(self, world)

    def create_qubit_at_station(self):
        new_qubit = Qubit(world=self.world, station=self)
        self.qubits += [new_qubit]
        return new_qubit


class Source(WorldObject):
    def __init__(self, world, position, target_stations):
        self.position = position
        self.target_stations = target_stations
        super(Source, self).__init__(self, world)

    def generate_pair(self):
        pass  # TODO
