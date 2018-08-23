import numpy as np


def get_arrays_info(arrays_shapes):
    shadpes = {
        'dogs_positions': (config.dogs_count, 2),
        'sheep_positions': (config.sheep_count, 2),
        'herd_centre': (2,),
        'observation': (config.dogs_count, config.rays_count),
        'dogs_rotations': (config.dogs_count,),
        'action': (config.dogs_count, 3)
    }
    arrays_sizes = _get_arrays_sizes(arrays_shapes)
    arrays_offsets = _get_arrays_offsets(arrays_sizes)
    arrays_total_size = _get_arrays_total_size(arrays_sizes)

    arrays = {}
    for key, _ in arrays_shapes.items():
        arrays[key] = {
            'shape': arrays_shapes[key],
            'size': arrays_sizes[key],
            'offset': arrays_offsets[key],
        }

    arrays_info = {
        'total_size': arrays_total_size,
        'arrays': arrays
    }

    return arrays_info


def _get_arrays_sizes(shapes):
    sizes = {}

    for key, value in shapes.items():
        sizes[key] = np.prod(np.array(list(value))) * 4

    return sizes


def _get_arrays_offsets(sizes):
    offset = 0
    info = {}

    for key, value in sizes.items():
        info[key] = offset
        offset += value

    return info


def _get_arrays_total_size(sizes):

    return sum(sizes.values())
