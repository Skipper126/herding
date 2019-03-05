import pyopencl as cl
import numpy as np


class SubBuffer:

    def __init__(self, ocl, buffer, origin, size):
        self.ocl = ocl
        self.buffer = buffer
        self.origin = origin
        self.size = size
        self.mapping = None

    def map_read(self):
        self.mapping = cl.enqueue_map_buffer(self.ocl.queue,
                                             self.buffer,
                                             cl.map_flags.READ,
                                             self.origin,
                                             self.size,
                                             np.float32)

    def map_write(self):
        self.mapping = cl.enqueue_map_buffer(self.ocl.queue,
                                             self.buffer,
                                             cl.map_flags.WRITE_INVALIDATE_REGION,
                                             self.origin,
                                             self.size,
                                             np.float32)

    def unmap(self):
        self.mapping.base.release()
