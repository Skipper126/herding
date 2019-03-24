from typing import NamedTuple, List, Tuple, Any


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


class ArrayInfo(NamedTuple):
    name: str
    shape: Tuple[int]
    size: int
    offset: int


class EnvData(NamedTuple):
    config: Config
    arrays_info: List[ArrayInfo]
    ocl: Any  # OpenCL type. "Any" to avoid circular dependency
