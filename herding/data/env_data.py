from dataclasses import dataclass
from herding import opencl


@dataclass
class Config:
    dogs_count: int
    sheep_count: int
    scan_radius: int
    neighbours_count: int
    agents_matrix_side_length: int
    neighbours_matrix_side_length: int
    agents_layout: str
    sheep_type: str
    herd_target_radius: int
    device: bool
    agent_radius: int
    movement_speed: int
    rotation_speed: int
    max_episode_reward: int
    time_penalty_rate: int
    sheep_flee_distance: int
    rays_count: int
    ray_length: int
    field_of_view: int
    agents_layout_width: int
    agents_layout_height: int
    window_width: int
    window_height: int
    channels_count: int
    seed: int
    target_x: int
    target_y: int


@dataclass
class Buffers:
    input_matrix: opencl.Buffer
    output_matrix: opencl.Buffer
    agents_matrix1: opencl.Buffer
    agents_matrix2: opencl.Buffer
    dogs_positions: opencl.Buffer
    sheep_positions: opencl.Buffer
    target_position: opencl.Buffer
    observation: opencl.Buffer
    rays_lengths: opencl.Buffer
    seed: opencl.Buffer
    reward: opencl.Buffer


@dataclass
class EnvData:
    config: Config = None
    shared_buffers: Buffers = None
    ocl: opencl.OpenCL = None
