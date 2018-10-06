import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda


def malloc(size: int) -> int:
    return cuda.mem_alloc(int(size))

def pagelocked_host_malloc(size, dtype) -> np.ndarray:
    return cuda.pagelocked_empty((int(size),), dtype)

def memcpy_htod(dest, src: np.ndarray):
    cuda.memcpy_htod(int(dest), src)

def memcpy_dtoh(dest: np.ndarray, src):
    cuda.memcpy_dtoh(dest, int(src))
