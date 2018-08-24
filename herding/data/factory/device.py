from herding import cuda
from herding.data.factory.info import get_arrays_info


def get_device_arrays(config):
    arrays_shapes = {
        'dogs_positions': (config.dogs_count, 2),
        'sheep_positions': (config.sheep_count, 2),
        'herd_centre': (2,),
        'observation': (config.dogs_count, config.rays_count),
        'dogs_rotations': (config.dogs_count,),
        'action': (config.dogs_count, 3)  # TODO improve syncing host_arrays and device_arrays with action
    }
    arrays_info = get_arrays_info(arrays_shapes)
    arrays_buffer = cuda.malloc(arrays_info['size'])

    device_arrays = {}
    for array, info in arrays_info['arrays'].items():
        if array is not 'action':
            device_arrays[array] = arrays_buffer + info['offset']

    return device_arrays
