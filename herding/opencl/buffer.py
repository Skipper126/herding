import pyopencl as cl
import numpy as np


class Buffer:

    def __init__(self, queue, context, shape, dtype, flags=None):
        self.queue = queue
        self.shape = shape
        self.dtype = dtype
        self.mapping = None
        self.buffer = cl.Buffer(context, cl.mem_flags.ALLOC_HOST_PTR | flags, np.prod(shape) * 4)

    def map_read(self) -> np.ndarray:
        self.mapping = cl.enqueue_map_buffer(self.queue,
                                             self.buffer,
                                             cl.map_flags.READ,
                                             0,
                                             self.shape,
                                             self.dtype)

        return self.mapping[0]

    def map_write(self) -> np.ndarray:
        self.mapping = cl.enqueue_map_buffer(self.queue,
                                             self.buffer,
                                             cl.map_flags.WRITE_INVALIDATE_REGION,
                                             0,
                                             self.shape,
                                             self.dtype)

        return self.mapping[0]

    def unmap(self):
        self.mapping[0].base.release()
