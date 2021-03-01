from typing import NamedTuple
from herding import data
import numpy as np


class EnvArrays(NamedTuple):
    dogs_positions: np.ndarray
    sheep_positions: np.ndarray
    target_position: np.ndarray
    observation: np.ndarray
    rays_lengths: np.ndarray


class EnvArraysMapper:

    def __init__(self, env_data: data.EnvData):
        self._buffers = [
            env_data.shared_buffers.dogs_positions,
            env_data.shared_buffers.sheep_positions,
            env_data.shared_buffers.target_position,
            env_data.shared_buffers.observation,
            env_data.shared_buffers.rays_lengths
        ]

    def map_env_arrays(self):
        return EnvArrays(*[buffer.map_read() for buffer in self._buffers])

    def unmap_env_arrays(self):
        for buffer in self._buffers:
            buffer.unmap()
