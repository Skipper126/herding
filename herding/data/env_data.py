from typing import NamedTuple, Dict, List
from pycuda.driver import DeviceAllocation
import numpy as np


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
    agents_layout_size: int
    skip_frames: int
    window_width: int
    window_height: int
    colors : Dict[str, List[float]]


class EnvData(NamedTuple):
    config: Config

    host_arrays: np.ndarray
    layout_arrays: np.ndarray
    reward_arrays: np.ndarray
    agents_controller_arrays: np.ndarray
    agents_controller_arrays_input

    dogs_positions: np.ndarray
    dogs_rotations: np.ndarray
    sheep_positions: np.ndarray
    target: np.ndarray
    observation: np.ndarray

    device_arrays: DeviceAllocation

    device_dogs_positions: int
    device_dogs_rotations: int
    device_sheep_positions: int
    device_target: int
    device_observation: int
    device_action: int
    device_rand_values: int

    device_rays_lengths: int
    device_rays_angles: int

