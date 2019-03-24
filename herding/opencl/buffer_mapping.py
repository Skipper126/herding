import pyopencl as cl
import numpy as np
from herding.data.env_data import EnvData, ArrayInfo
from typing import List


class BufferMapping:

    def __init__(self, ocl, offset, shape):
        self.ocl = ocl
        self.buffer = ocl.buffer
        self.offset = offset
        self.shape = tuple(shape)
        self.mapping = None

    def map_read(self):
        self.mapping = cl.enqueue_map_buffer(self.ocl.queue,
                                             self.buffer,
                                             cl.map_flags.READ,
                                             self.offset,
                                             self.shape,
                                             np.float32)

        return self.mapping[0]

    def map_write(self):
        self.mapping = cl.enqueue_map_buffer(self.ocl.queue,
                                             self.buffer,
                                             cl.map_flags.WRITE_INVALIDATE_REGION,
                                             self.offset,
                                             self.shape,
                                             np.float32)

        return self.mapping[0]

    def unmap(self):
        self.mapping[0].base.release()


def create_buffer_mapping(env_data: EnvData, array_name: str) -> BufferMapping:
    array_info = _get_array_info(env_data, array_name)
    return BufferMapping(env_data.ocl, array_info.offset, array_info.shape)


class MultipleBufferMapping:

    def __init__(self, ocl, total_offset, total_size, offsets, shapes, sizes):
        self.ocl = ocl
        self.buffer = ocl.buffer
        self.total_offset = total_offset
        self.total_size = total_size
        self.offsets = offsets
        self.shapes = shapes
        self.sizes = sizes
        self.arrays_count = len(self.sizes)
        self.mapping = None

    def map_read(self):
        self.mapping = cl.enqueue_map_buffer(self.ocl.queue,
                                             self.buffer,
                                             cl.map_flags.READ,
                                             self.total_offset,
                                             self.total_size,
                                             np.float32,
                                             is_blocking=False)
        arrays = self._create_arrays()
        self.mapping[1].wait()
        return arrays

    def map_write(self):
        self.mapping = cl.enqueue_map_buffer(self.ocl.queue,
                                             self.buffer,
                                             cl.map_flags.WRITE_INVALIDATE_REGION,
                                             self.total_offset,
                                             self.total_size,
                                             np.float32,
                                             is_blocking=False)

        arrays = self._create_arrays()
        self.mapping[1].wait()
        return arrays

    def unmap(self):
        self.mapping[0].base.release()

    def _create_arrays(self):
        arrays = []
        for i in range(self.arrays_count):
            arrays.append(
                np.frombuffer(self.mapping,
                              np.float32,
                              self.sizes[i],
                              self.offsets[i]).reshape(self.shapes[i]))

        return tuple(arrays)


def create_multiple_buffer_mapping(env_data: EnvData, arrays_names: List[str]) -> MultipleBufferMapping:
    total_offset = 0
    total_size = 0
    offsets = []
    shapes = []
    sizes = []

    for i, name in enumerate(arrays_names):
        array_info = _get_array_info(env_data, name)
        if i == 0:
            total_offset = array_info.offset
        total_size += array_info.size
        offsets.append(array_info.offset)
        shapes.append(array_info.shape)
        sizes.append(array_info.size)

    return MultipleBufferMapping(env_data.ocl,
                                 total_offset,
                                 total_size,
                                 offsets,
                                 shapes,
                                 sizes)


def _get_array_info(env_data, array_name) -> ArrayInfo:
    return [info for info in env_data.arrays_info if info.name == array_name][0]
