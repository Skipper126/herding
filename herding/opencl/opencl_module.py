import pyopencl as cl
from pathlib import Path
from herding.data.configuration import get_ocl_defines
from herding.data.env_data import EnvData


class OpenClModule:

    def __init__(self, ocl, prg, function):
        self.queue = ocl.queue
        self.prg = prg
        self.kernel = getattr(prg, function)
        self.kernel.set_arg(0, ocl.buffer)

    def run(self, nd_range):
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, (nd_range,), (nd_range,))


def create_module(env_data: EnvData, file: str, function: str) -> OpenClModule:
    project_root = _get_project_root_path()
    with open(project_root + file, 'r') as f:
        source = f.read()

    options = ['-I ' + project_root]
    options.extend(get_ocl_defines(env_data))
    prg = cl.Program(env_data.ocl.context, source).build(
        options=options
    )

    return OpenClModule(env_data.ocl, prg, function)


def _get_project_root_path():
    file_path = Path(__file__)
    # assume current file path is herding/opencl/opencl_module.py
    root_path = file_path.parent.parent.parent
    return str(root_path) + '/'
