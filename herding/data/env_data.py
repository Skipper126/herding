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
    skip_frames: str
    window_width: int
    window_height: int
    agent_geom_resolution: int

class EnvData(NamedTuple):
    config: Config
    host_arrays: np.ndarray
    dogs_positions: np.ndarray
    sheep_positions: np.ndarray
    rays: np.array
    dogs_rotations: np.ndarray
    herd_centre: np.ndarray
    observation: np.ndarray

    device_arrays: DeviceAllocation
    device_dogs_positions: int
    device_sheep_positions: int
    device_rays: int
    device_dogs_rotations: int
    device_action: int
