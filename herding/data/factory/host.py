import numpy as np
from herding.data.factory.info import get_arrays_info


def get_host_arrays(arrays_shapes):
    shared_arrays_info = get_arrays_info(arrays_shapes['shared'])
    shared_arrays_buffer = np.empty((shared_arrays_info['total_size'],), dtype=np.float32)

    host_arrays_info = get_arrays_info(arrays_shapes['host'])
    host_arrays_buffer = np.empty((shared_arrays_info['total_size'],), dtype=np.float32)

    host_arrays = {
        'host_arrays': shared_arrays_buffer,
        **_get_arrays(shared_arrays_info['arrays'], shared_arrays_buffer),
        **_get_arrays(host_arrays_info['arrays'], host_arrays_buffer)
    }

    return host_arrays


def _get_arrays(info, buffer):
    arrays = {}
    for array, info in info.items():
        if array is not 'action':
            arrays[array] = np.frombuffer(buffer,
                         dtype=np.float32,
                         offset=info['offset'] * 4,
                         count=info['size']).reshape(info['shape'])

    return arrays
