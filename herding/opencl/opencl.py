import pyopencl as cl
from typing import NamedTuple


class OpenCL(NamedTuple):
    context: cl.Context
    queue: cl.CommandQueue
    buffer: cl.Buffer


def create_opencl(buffer_size):
    ctx = cl.create_some_context(answers=[0, 0])
    queue = cl.CommandQueue(ctx)
    buffer = cl.Buffer(ctx, cl.mem_flags.ALLOC_HOST_PTR, buffer_size * 4)

    return OpenCL(ctx, queue, buffer)
