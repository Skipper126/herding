from herding import cuda
from herding.data.factory.info import get_arrays_info


def get_device_arrays(config):
    arrays_shapes = {
        'device_dogs_positions': (config.dogs_count, 2),
        'device_sheep_positions': (config.sheep_count, 2),
        'device_herd_centre': (2,),
        'device_observation': (config.dogs_count, config.rays_count, 2),
        'device_dogs_rotations': (config.dogs_count,),
        'device_action': (config.dogs_count, 3)  # TODO improve syncing host_arrays and device_arrays with action
    }
    arrays_info = get_arrays_info(arrays_shapes)
    arrays_device_allocation = cuda.malloc(arrays_info['total_size'] * 4)
    arrays_buffer = int(arrays_device_allocation)

    device_arrays = {
        'device_arrays': arrays_device_allocation
    }
    for array, info in arrays_info['arrays'].items():
            device_arrays[array] = arrays_buffer + (info['offset'] * 4)

    return device_arrays
