from herding import cuda
from herding.data.factory.info import get_arrays_info


def get_device_arrays(arrays_shapes):
    device_arrays_shapes = {
        **arrays_shapes['shared'],
        **arrays_shapes['device']
    }
    arrays_info = get_arrays_info(device_arrays_shapes)
    arrays_device_allocation = cuda.malloc(arrays_info['total_size'] * 4)
    arrays_buffer = int(arrays_device_allocation)

    device_arrays = {
        'device_arrays': arrays_device_allocation
    }
    for array, info in arrays_info['arrays'].items():
            device_arrays['device_' + array] = arrays_buffer + (info['offset'] * 4)

    return device_arrays
