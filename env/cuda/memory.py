import numpy as np
from ctypes import c_longlong


def malloc(size: int) -> int:
    return 0


def memcpy_htod(dest: c_longlong, src: np.ndarray):
    pass


def memcpy_dtoh(dest: np.ndarray, src: c_longlong):
    pass
