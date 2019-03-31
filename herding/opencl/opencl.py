import pyopencl as cl
from typing import Dict, Tuple, List
import numpy as np
from herding.opencl.buffer import Buffer
from herding.opencl.module import Module
from pathlib import Path


class OpenCL:

    def __init__(self, definitions: Dict[str, int]):
        self.context = cl.create_some_context(answers=[0, 0])
        self.queue = cl.CommandQueue(self.context)
        self.project_root = _get_project_root_path()
        self.options = _convert_definitions(definitions)
        self.options.append('-I ' + self.project_root)

    def create_buffer(self, shape: Tuple, dtype) -> Buffer:
        return Buffer(self.queue, self.context, shape, dtype)

    def create_module(self, file: str, function: str, args: List[Buffer]) -> Module:
        with open(self.project_root + file, 'r') as f:
            source = f.read()

        prg = cl.Program(self.context, source).build(
            options=self.options
        )
        buffers = [arg.buffer for arg in args]

        return Module(self.queue, prg, function, buffers)


def _get_project_root_path():
    file_path = Path(__file__)
    # assume current file path is herding/opencl/opencl.py
    root_path = file_path.parent.parent.parent
    return str(root_path) + '/'


def _convert_definitions(definitions):
    options = ['-D ' + name.upper() + '=' + str(value) for name, value in definitions.items()]

    return options
