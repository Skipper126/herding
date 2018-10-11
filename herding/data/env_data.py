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
    reward_type: str
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
    channels_count: int

class HostArrays(NamedTuple):
    rays_lengths: np.ndarray
    dogs_positions: np.ndarray
    dogs_rotations: np.ndarray
    sheep_positions: np.ndarray
    target: np.ndarray
    observation: np.ndarray
    action: np.ndarray

class DeviceArrays(NamedTuple):
    rays_lengths: int
    dogs_positions: int
    dogs_rotations: int
    sheep_positions: int
    target: int
    observation: int
    action: int

class EnvData(NamedTuple):
    config: Config
    host_buffer: np.ndarray
    device_buffer: DeviceAllocation
    host_arrays: HostArrays
    device_arrays: DeviceArrays
