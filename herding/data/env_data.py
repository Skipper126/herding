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
    rays_count: int
    ray_length: int
    field_of_view: int
    agents_layout_range: int
    skip_frames: int
    window_width: int
    window_height: int
    channels_count: int
    seed: int


class Buffers(NamedTuple):
    rays_lengths: opencl.Buffer
    dogs_positions: opencl.Buffer
    dogs_rotations: opencl.Buffer
    sheep_positions: opencl.Buffer
    target_position: opencl.Buffer
    observation: opencl.Buffer
    action: opencl.Buffer
    seed: opencl.Buffer
    common_output: opencl.Buffer


class EnvData(NamedTuple):
    config: Config
    buffers: Buffers
    ocl: opencl.OpenCL
