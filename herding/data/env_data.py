from typing import NamedTuple
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
    dog_color_r: int
    dog_color_g: int
    dog_color_b: int
    sheep_color_r: int
    sheep_color_g: int
    sheep_color_b: int
    target_color_r: int
    target_color_g: int
    target_color_b: int
    dog_bark_color_r: int
    dog_bark_color_g: int
    dog_bark_color_b: int

class EnvData(NamedTuple):
    config: Config
    host_arrays: np.ndarray
    dogs_positions: np.ndarray
    sheep_positions: np.ndarray
    observation: np.ndarray
    rays_lengths: np.ndarray
    dogs_rotations: np.ndarray
    rand_values: np.ndarray
    target: np.ndarray

    device_arrays: DeviceAllocation
    device_dogs_positions: int
    device_sheep_positions: int
    device_observation: int
    device_rays_lengths: int
    device_dogs_rotations: int
    device_rand_values: int
    device_target: int
    device_action: int