import pyopencl as cl
from typing import List
from herding.opencl import OpenCL


class OpenClModule:

    def __init__(self, ocl, prg, buffer):
        self.queue = ocl.queue
        self.prg = prg
        # assume only one kernel per module
        self.kernel = prg.all_kernels()[0]
        self.kernel.set_arg(0, buffer)

    def run(self, nd_range):
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, nd_range, nd_range)


def create_opencl_module(ocl: OpenCL, file: str, include_paths: List[str]) -> OpenClModule:
    with open(file, 'r') as f:
        source = f.read()

    prg = cl.Program(ocl.context, source).build(
        options=['-I ' + path for path in include_paths]
    )
    return OpenClModule(ocl, prg, ocl.buffer)
