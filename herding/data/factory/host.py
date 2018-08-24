import numpy as np
from herding.data.factory.info import get_arrays_info


def get_host_arrays(config):
    arrays_shapes = {
        'dogs_positions': (config.dogs_count, 2),
        'sheep_positions': (config.sheep_count, 2),
        'herd_centre': (2,),
        'observation': (config.dogs_count, config.rays_count, 2),
        'dogs_rotations': (config.dogs_count,)
    }
    arrays_info = get_arrays_info(arrays_shapes)
    arrays_buffer = np.empty((arrays_info['total_size'],), dtype=np.float32)

    host_arrays = {
        'host_arrays': arrays_buffer
    }
    for array, info in arrays_info['arrays'].items():
        if array is not 'action':
            host_arrays[array] = _get_array(info, arrays_buffer)

    return host_arrays


def _get_array(info, buffer):
    return np.frombuffer(buffer,
                         dtype=np.float32,
                         offset=info['offset'] * 4,
                         count=info['size']).reshape(info['shape'])
