from typing import NamedTuple
from herding import opencl


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
    sheep_flee_distance: int
    rays_count: int
    ray_length: int
    field_of_view: int
    agents_layout_width: int
    agents_layout_height: int
    skip_frames: int
    window_width: int
    window_height: int
    channels_count: int
    seed: int


class Buffers(NamedTuple):
    dogs_positions: opencl.Buffer
    sheep_positions: opencl.Buffer
    target_position: opencl.Buffer
    observation: opencl.Buffer
    rays_lengths: opencl.Buffer
    seed: opencl.Buffer


class EnvData(NamedTuple):
    config: Config
    shared_buffers: Buffers
    ocl: opencl.OpenCL
