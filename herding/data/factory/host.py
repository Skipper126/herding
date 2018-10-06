import numpy as np
from herding.data.factory.info import get_arrays_info
from herding.data.env_data import HostArrays
from herding import cuda


def get_host_arrays(arrays_shapes):
    arrays_info = get_arrays_info({
        **arrays_shapes['shared'],
        **arrays_shapes['host']
    })
    total_size = sum(array_info.size for array_info in arrays_info)
    arrays_buffer = cuda.pagelocked_host_malloc(total_size, np.float32)

    host_arrays = HostArrays(**_get_arrays(arrays_buffer, arrays_info))
    return arrays_buffer, host_arrays


def _get_arrays(buffer, arrays_info):
    arrays = {}
    for array_info in arrays_info:
        arrays[array_info.name] = \
            np.frombuffer(buffer,
                          dtype=np.float32,
                          offset=array_info.offset,
                          count=array_info.size).reshape(array_info.shape)

    return arrays
