from dataclasses import dataclass
from herding import opencl


@dataclass
class Config:
    dogs_count: int
    sheep_count: int
    agents_layout: str
    sheep_type: str
    herd_target_radius: int
    reward_type: str
    device: str
    agent_radius: int
    max_movement_speed: int
    max_episode_reward: int
    sheep_flee_distance: int
    rays_count: int
    ray_length: int
    agents_layout_width: int
    agents_layout_height: int
    skip_frames: int
    window_width: int
    window_height: int
    channels_count: int
    seed: int


@dataclass
class Buffers:
    dogs_positions: opencl.Buffer
    sheep_positions: opencl.Buffer
    target_position: opencl.Buffer
    observation: opencl.Buffer
    rays_lengths: opencl.Buffer
    seed: opencl.Buffer


@dataclass
class EnvData:
    config: Config = None
    shared_buffers: Buffers = None
    ocl: opencl.OpenCL = None
