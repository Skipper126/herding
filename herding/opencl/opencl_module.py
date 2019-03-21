import pyopencl as cl
from herding.opencl import OpenCL
from pathlib import Path


class OpenClModule:

    def __init__(self, ocl, prg, function):
        self.queue = ocl.queue
        self.prg = prg
        self.kernel = getattr(prg, function)
        self.kernel.set_arg(0, ocl.buffer)

    def run(self, nd_range):
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, nd_range, nd_range)


def create_module(ocl: OpenCL, file: str, function: str) -> OpenClModule:
    project_root = _get_project_root_path()
    with open(project_root + file, 'r') as f:
        source = f.read()

    prg = cl.Program(ocl.context, source).build(
        options=['-I ' + project_root]
    )

    return OpenClModule(ocl, prg, function)


def _get_project_root_path():
    file_path = Path(__file__)
    # assume current file path is herding/opencl/opencl_module.py
    return file_path.parent.parent.parent + '/'
