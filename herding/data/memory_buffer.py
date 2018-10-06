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



    env_data_names = _get_arrays_names_list_from_env_data(env_data)
    array_index_in_env_data = _get_env_data_array_index(env_data_names, arrays_names[0])

    if _check_arrays_set(env_data_names, arrays_names, array_index_in_env_data) is False:
        raise Exception('Arrays don\'t fit env_data.')

    end_index = array_index_in_env_data + len(arrays_names) - 1
    offset, count = _get_offset_and_count(env_data, array_index_in_env_data, end_index)

    host_buffer = np.frombuffer(env_data.host_buffer, np.float32, count, offset)
    device_buffer = getattr(env_data, 'device_' + arrays_names[-1])
    buffer = MemoryBuffer(host_buffer, device_buffer)

    return buffer

def _get_offset_and_count(env_data, start_index, end_index):
    offset = 0
    count = 0
    index = 0
    for _, array in env_data._asdict().items():
        if type(array) is not np.ndarray:
            continue
        if index < start_index:
            offset += array.size * 4
        else:
            count += array.size
        index += 1
        if index == end_index:
            break

    return offset, count


def _get_arrays_from_names(env_data, packet_names):
    arrays = []
    for packet_name in packet_names:
        arrays.append(getattr(env_data, packet_name))

    return arrays


def _check_arrays_set(env_data_names, packet_names, array_index_in_env_data):
    fits = True
    for packet_name in packet_names:
        if packet_name != env_data_names[array_index_in_env_data]:
            fits = False
            break
        else:
            array_index_in_env_data += 1

    return fits


def _get_env_data_array_index(env_data_names, array_name):
    index = -1
    for i in range(len(env_data_names)):
        if env_data_names[i] == array_name:
            index = i
            break
    if index == -1:
        raise Exception('Array name not found in env_data.')

    return index


def _get_arrays_names_list_from_env_data(env_data):
    arrays_names = []
    for key, value in env_data._asdict().items():
        arrays_names.append(key)
    return arrays_names
