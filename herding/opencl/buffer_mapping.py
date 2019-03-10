import pyopencl as cl
import numpy as np
from herding.data import EnvData
from typing import List


class BufferMapping:

    def __init__(self, ocl, buffer, origin, size):
        self.ocl = ocl
        self.buffer = buffer
        self.origin = origin
        self.size = size
        self.mapping = None

    def map_read(self):
        self.mapping = cl.enqueue_map_buffer(self.ocl.queue,
                                             self.buffer,
                                             cl.map_flags.READ,
                                             self.origin,
                                             self.size,
                                             np.float32)

    def map_write(self):
        self.mapping = cl.enqueue_map_buffer(self.ocl.queue,
                                             self.buffer,
                                             cl.map_flags.WRITE_INVALIDATE_REGION,
                                             self.origin,
                                             self.size,
                                             np.float32)

    def unmap(self):
        self.mapping.base.release()


def create_buffer_mapping(env_data: EnvData, arrays_names: List[str]) -> BufferMapping:
    arrays_keys, \
     arrays_values = (lambda dict: (list(dict.keys()), list(dict.values())))(env_data.arrays._asdict())

    indexes = []
    for name in arrays_names:
        indexes.append(arrays_keys.index(name))
    indexes.sort()

    if indexes != list(range(indexes[0], indexes[-1] + 1)):
        raise Exception('Arrays don\'t match host arrays.')

    size = sum(arrays_values[index].size for index in indexes)
    offset = sum(arrays_values[index].size for index in range(indexes[0])) * 4

    return BufferMapping(env_data.ocl, env_data.ocl.buffer, offset, size)
