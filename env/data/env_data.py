from typing import NamedTuple
import numpy as np
from ctypes import c_longlong


class Config(NamedTuple):
    dogs_count: int
    sheep_count: int
    agents_layout: str
    sheep_type: str
    herd_target_radius: int
    rotation_mode: str
    agent_radius: int
    max_movement_speed: int
    max_rotation_speed: int
    max_episode_reward: int
    rays_count: int
    ray_length: int
    field_of_view: int
    map_height: int
    map_width: int
    hardware_acceleration: str


class EnvData(NamedTuple):
    config: Config
    host_env_data: np.ndarray
    dogs_positions: np.ndarray
    sheep_positions: np.ndarray
    herd_centre: np.ndarray
    observation: np.ndarray
    dogs_rotations: np.ndarray

    gpu_env_data: c_longlong
    gpu_config: c_longlong
    gpu_dogs_positions: c_longlong
    gpu_sheep_positions: c_longlong
    gpu_herd_centre: c_longlong
    gpu_observation: c_longlong
    gpu_dogs_rotations: c_longlong
    gpu_action: c_longlong
