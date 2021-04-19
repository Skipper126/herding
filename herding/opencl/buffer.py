import pyopencl as cl
import numpy as np


class Buffer:

    def __init__(self, queue, context, shape, dtype):
        self.queue = queue
        self.shape = shape
        self.dtype = dtype
        self.mapping = None
        self.map_count = 0
        self.buffer = cl.Buffer(context, cl.mem_flags.ALLOC_HOST_PTR,
                                np.prod(shape).astype(np.int32) * np.dtype(self.dtype).itemsize)

    def map_read(self) -> np.ndarray:
        if self.mapping is None:
            self.mapping = cl.enqueue_map_buffer(self.queue,
                                                 self.buffer,
                                                 cl.map_flags.READ,
                                                 0,
                                                 self.shape,
                                                 self.dtype)
        self.map_count += 1
        return self.mapping[0]

    def map_write(self) -> np.ndarray:
        if self.mapping is None:
            self.mapping = cl.enqueue_map_buffer(self.queue,
                                                 self.buffer,
                                                 cl.map_flags.WRITE_INVALIDATE_REGION,
                                                 0,
                                                 self.shape,
                                                 self.dtype)
        self.map_count += 1
        return self.mapping[0]

    def unmap(self):
        self.map_count -= 1
        if self.map_count == 0:
            self.mapping[0].base.release()
            self.mapping = None
