from typing import NamedTuple
from env.data.config import Config
import numpy as np
from ctypes import c_longlong


class EnvData(NamedTuple):
    config: Config
    dogs_positions: np.ndarray
    sheep_positions: np.ndarray
    herd_centre: np.ndarray
    observation: np.ndarray
    dogs_rotations: np.ndarray

    gpu_config: c_longlong
    gpu_dogs_positions: c_longlong
    gpu_sheep_positions: c_longlong
    gpu_herd_centre: c_longlong
    gpu_observation: c_longlong
    gpu_dogs_rotations: c_longlong
    gpu_action: c_longlong
