import numpy as np


"""
arrays_info: {
    total_size,
    arrays[array_name]: {
        shape,
        size,
        offset
    }
}
"""


def get_arrays_info(arrays_shapes):
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
        sizes[key] = np.prod(np.array(list(value)))

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
