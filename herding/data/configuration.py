from typing import Dict, List, Tuple, NamedTuple, Any
from herding.data.env_data import Config
import numpy as np


class BufferInfo(NamedTuple):
    name: str
    shape: Tuple
    dtype: Any


def get_shared_buffers_info(config: Config) -> List[BufferInfo]:
    return [
        BufferInfo(
            name='dogs_positions',
            shape=(config.dogs_count, 3),
            dtype=np.float32
        ),
        BufferInfo(
            name='sheep_positions',
            shape=(config.sheep_count, 2),
            dtype=np.float32
        ),
        BufferInfo(
            name='target_position',
            shape=(2,),
            dtype=np.float32
        ),
        BufferInfo(
            name='observation',
            shape=(config.dogs_count, config.rays_count, 3),
            dtype=np.float32
        ),
        BufferInfo(
            name='rays_lengths',
            shape=(config.dogs_count, config.rays_count),
            dtype=np.float32
        ),
        BufferInfo(
            name='seed',
            shape=(config.dogs_count + config.sheep_count + 1,),
            dtype=np.int32
        )
    ]


def get_default_configuration() -> Dict:
    return {
        "dogs_count": 1,
        "sheep_count": 3,
        "agents_layout": "random",
        "sheep_type": "simple",
        "herd_target_radius": 100,
        "rotation_mode": "free",
        "reward_type": "furthest_sheep",
        "agent_radius": 10,
        "max_movement_speed": 5,
        "max_rotation_speed": 90,
        "max_episode_reward": 100,
        "rays_count": 128,
        "ray_length": 300,
        "field_of_view": 180,
        "agents_layout_range": 800,
        "skip_frames": 1,
        "window_width": 1000,
        "window_height": 800,
        "channels_count": 3,
        "seed": 100
    }


def get_kernel_definitions(config: Config) -> Dict[str, int]:
    names = [
        'dogs_count',
        'sheep_count',
        'herd_target_radius',
        'agent_radius',
        'max_movement_speed',
        'max_rotation_speed',
        'max_episode_reward',
        'rays_count',
        'ray_length',
        'field_of_view',
        'agents_layout_range',
        'channels_count'
    ]
    config_dict = config._asdict()
    return dict((name, config_dict[name]) for name in names if name in config_dict)