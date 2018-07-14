import ctypes as c
import numpy as np
import multiprocessing as mp


class SharedNumpy:

    def __init__(self, shape):
        self._value = mp.Array(c.c_double, np.prod(np.array(list(shape))), lock=False)
        self._shape = shape

    def get(self):
        return np.frombuffer(self._value).reshape(self._shape)
