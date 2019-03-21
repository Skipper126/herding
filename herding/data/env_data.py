from typing import NamedTuple, List, Tuple
from herding.data import configuration
from herding import opencl
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
    ocl: opencl.OpenCL


def create_env_data(params):
    config = _create_config(params)
    arrays_info = _get_arrays_info(config)
    buffer_size = sum(array_info.size for array_info in arrays_info)
    ocl = opencl.create_opencl(buffer_size)

    env_data = EnvData(config, arrays_info, ocl)
    _init_seed(env_data)
    return env_data


def _create_config(params):
    config_dict = configuration.get_default_configuration()
    config_dict.update(params)
    config = Config(**config_dict)

    return config


def _get_arrays_info(config):
    arrays_shapes = configuration.get_arrays_shapes(config)
    arrays_info = []
    offset = 0
    for name, shape in arrays_shapes.items():
        size = int(np.prod(np.array(list(shape))))
        arrays_info.append(ArrayInfo(
            name=name,
            shape=shape,
            size=size,
            offset=offset,
        ))
        offset += size

    return arrays_info


def _init_seed(env_data):
    seed_mapping = opencl.create_buffer_mapping(env_data, 'seed')
    seed_array = seed_mapping.map_write()
    rand_array = np.random.randint(0, 2147483647)
    np.copyto(seed_array, rand_array)
    seed_mapping.unmap()
