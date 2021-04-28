from typing import NamedTuple
from herding import data
import numpy as np


class EnvArrays(NamedTuple):
    current_agents_matrix: np.ndarray
    observation: np.ndarray


class EnvArraysMapper:

    def __init__(self, env_data: data.EnvData):
        self._buffers = [
            env_data.shared_buffers.current_agents_matrix,
            env_data.shared_buffers.observation
        ]

    def map_env_arrays(self):
        return EnvArrays(*[buffer.map_read() for buffer in self._buffers])

    def unmap_env_arrays(self):
        for buffer in self._buffers:
            buffer.unmap()
