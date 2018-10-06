from herding import cuda
from herding.data.factory.info import get_arrays_info


def get_device_arrays(arrays_shapes):
    arrays_info = get_arrays_info({
        **arrays_shapes['shared'],
        **arrays_shapes['device']
    })
    total_size = sum(array_info.size for array_info in arrays_info)
    arrays_device_allocation = cuda.malloc(total_size * 4)
    arrays_buffer = int(arrays_device_allocation)

    device_arrays = {}

    for array_info in arrays_info:
            device_arrays[array_info.name] = arrays_buffer + (array_info.offset)

    return arrays_device_allocation, device_arrays
