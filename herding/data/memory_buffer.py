import numpy as np
from typing import List
from herding.data import EnvData
from herding import cuda

class MemoryBuffer:

    def __init__(self, host_buffer, device_buffer):
        self._host_buffer = host_buffer
        self._device_buffer = device_buffer

    def sync_dtoh(self):
        cuda.memcpy_dtoh(self._host_buffer, self._device_buffer)

    def sync_htod(self):
        cuda.memcpy_htod(self._device_buffer, self._host_buffer)


def get_memory_buffer(env_data: EnvData, arrays_names: List[str]) -> MemoryBuffer:

    host_arrays_keys, host_arrays_values = (lambda dict : (list(dict.keys()), list(dict.values())))\
                                           (env_data.host_arrays._asdict())
    device_arrays = env_data.device_arrays._asdict()

    indexes = []
    for name in arrays_names:
        indexes.append(host_arrays_keys.index(name))
    indexes.sort()

    if indexes != list(range(indexes[0],indexes[-1] + 1)):
        raise Exception('Arrays don\'t match host arrays.')

    size = sum((host_arrays_values)[index].size for index in indexes)
    offset = sum((host_arrays_values)[index].size for index in range(indexes[0])) * 4

    host_buffer = np.frombuffer(env_data.host_buffer, np.float32, size, offset)
    device_buffer = device_arrays[arrays_names[0]]
    buffer = MemoryBuffer(host_buffer, device_buffer)

    return buffer
