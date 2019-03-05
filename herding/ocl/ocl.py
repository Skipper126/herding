import pyopencl as cl
from herding.ocl import OpenClModule
from herding.ocl import SubBuffer


class OpenCL:

    def __init__(self):
        self.ctx = cl.create_some_context(answers=[0, 0])
        self.queue = cl.CommandQueue(self.ctx)

    def compile(self, file, include_paths) -> OpenClModule:
        pass

    def create_buffer(self, size: int) -> cl.Buffer:
        return cl.Buffer(self.ctx, cl.mem_flags.ALLOC_HOST_PTR, size)

    def get_sub_buffer(self, buffer: cl.Buffer, origin: int, size: int) -> SubBuffer:
        return SubBuffer(self, buffer, origin, size)
