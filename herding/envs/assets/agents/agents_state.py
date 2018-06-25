import ctypes as c
import numpy as np
import multiprocessing as mp


class AgentsSharedState:

    def __init__(self, env):
        self.dogs_positions = mp.Array(c.c_double, env.dog_count * 2, lock=False)
        self.sheep_positions = mp.Array(c.c_double, env.sheep_count * 2, lock=False)

        self.dogs_done = mp.Array(c.c_bool, env.dog_count, lock=False)
        self.sheep_done = mp.Array(c.c_bool, env.sheep_count, lock=False)

    def get_dogs_positions(self):
        return self.get_linked_numpy_array(self.dogs_positions, ())

    @staticmethod
    def get_linked_numpy_array(source_array, dim):
        return np.frombuffer(source_array).reshape(dim)
