from typing import NamedTuple
from herding import data
import numpy as np


class EnvArrays(NamedTuple):
    input_matrix: np.ndarray
    observation: np.ndarray


class EnvArraysMapper:

    def __init__(self, env_data: data.EnvData):
        self.shared_buffers = env_data.shared_buffers
        self._buffers = []

    def map_env_arrays(self):
        self._buffers = []
        self._buffers = [
            self.shared_buffers.output_matrix,
            self.shared_buffers.observation
        ]
        return EnvArrays(*[buffer.map_read() for buffer in self._buffers])

    def unmap_env_arrays(self):
        for buffer in self._buffers:
            buffer.unmap()
