import dataclasses
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
            dtype=np.uint32
        )
    ]


def get_default_configuration() -> Dict:
    return {
        'dogs_count': 1,
        'sheep_count': 3,
        'agents_layout': 'random',
        'sheep_type': 'simple',
        'rotation_mode': 'free',
        'reward_type': 'medium_distance',
        'device': 'gpu',
    }


def get_internal_configuration() -> Dict:
    return {
        'herd_target_radius': 200,
        'agent_radius': 10,
        'movement_speed': 5,
        'rotation_speed': 5,  # In degrees
        'max_episode_reward': 100,
        'time_penalty_rate': 0.01,
        'sheep_flee_distance': 300,
        'rays_count': 128,
        'ray_length': 500,
        'field_of_view': 180,
        'agents_layout_width': 900,
        'agents_layout_height': 600,
        'window_width': 1000,
        'window_height': 800,
        'channels_count': 3,
        'seed': None
    }


def get_kernel_definitions(config: Config) -> Dict[str, int]:
    names = [
        'dogs_count',
        'sheep_count',
        'herd_target_radius',
        'agent_radius',
        'movement_speed',
        'rotation_speed',
        'max_episode_reward',
        'sheep_flee_distance',
        'rays_count',
        'ray_length',
        'field_of_view',
        'agents_layout_width',
        'agents_layout_height',
        'channels_count'
    ]
    config_dict = dataclasses.asdict(config)
    return dict((name, config_dict[name]) for name in names if name in config_dict)
