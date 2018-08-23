import numpy as np
from env.data.factory.info import get_arrays_info


def get_host_arrays(config):
    arrays_shapes = {
        'dogs_positions': (config.dogs_count, 2),
        'sheep_positions': (config.sheep_count, 2),
        'herd_centre': (2,),
        'observation': (config.dogs_count, config.rays_count),
        'dogs_rotations': (config.dogs_count,)
    }
    arrays_info = get_arrays_info(arrays_shapes)
    arrays_buffer = np.empty((arrays_info['size'],), dtype=np.byte)

    host_arrays = {}
    for array, info in arrays_info['arrays'].items():
        if array is not 'action':
            host_arrays[array] = _get_array(info, arrays_buffer)

    return host_arrays


def _get_array(info, buffer):
    return np.frombuffer(buffer,
                         dtype=np.float32,
                         offset=info['offset'],
                         count=info['size'] / 4).reshape(info['shape'])
