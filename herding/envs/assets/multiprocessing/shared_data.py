import ctypes as c
import numpy as np
import multiprocessing as mp


class SharedNumpy:

    def __init__(self, shape):
        self._value = mp.Array(c.c_double, np.prod(np.array(list(shape))), lock=False)
        self._shape = shape

    def get(self):
        return np.frombuffer(self._value).reshape(self._shape)


class SharedData:

    def __init__(self, env):

        self._dogs_positions = SharedNumpy((env.dog_count, 2))
        self._sheep_positions = SharedNumpy((env.sheep_count, 2))
        self._herd_centre = SharedNumpy((2,))

        self._dogs_done = SharedNumpy((env.dog_count,))
        self._sheep_done = SharedNumpy((env.sheep_count,))

    def get_variable(self, name):
        return getattr(self, name).get()
