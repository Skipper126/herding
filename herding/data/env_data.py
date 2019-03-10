from typing import NamedTuple, List
from herding.data import configuration
from herding import opencl
import pyopencl as cl
import numpy as np
import os


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
    agents_layout_size: int
    skip_frames: int
    window_width: int
    window_height: int
    channels_count: int


class Arrays(NamedTuple):
    rays_lengths: np.ndarray
    dogs_positions: np.ndarray
    dogs_rotations: np.ndarray
    sheep_positions: np.ndarray
    target: np.ndarray
    observation: np.ndarray
    action: np.ndarray


class EnvData(NamedTuple):
    config: Config
    arrays: Arrays
    ocl: opencl.OpenCL


class ArrayInfo(NamedTuple):
    name: str
    shape: List[int]
    size: int
    offset: int


def get_env_data_header_path() -> str:
    path = os.path.join(os.path.dirname(__file__), 'env_data.h')

    return path


def create_env_data(params):
    config = _create_config(params)
    arrays_info = _get_arrays_info(config)
    buffer_size = sum(array_info.size for array_info in arrays_info)
    ocl = opencl.create_opencl(buffer_size)
    arrays = _create_arrays(arrays_info, ocl, buffer_size)

    return EnvData(config, arrays, ocl)


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


def _create_arrays(arrays_info, ocl, buffer_size):
    host_buffer = _get_host_buffer(ocl, buffer_size)

    arrays = {}
    for array_info in arrays_info:
        arrays[array_info.name] = \
            np.frombuffer(host_buffer,
                          dtype=np.float32,
                          offset=array_info.offset * 4,
                          count=array_info.size).reshape(array_info.shape)

    return Arrays(**arrays)


# TODO explain how this works
def _get_host_buffer(ocl, buffer_size):
    buffer_mapping = cl.enqueue_map_buffer(ocl.queue,
                                           ocl.buffer,
                                           cl.map_flags.READ,
                                           0,
                                           buffer_size,
                                           np.float32)
    host_buffer = np.frombuffer(buffer_mapping, np.float32)
    buffer_mapping.base.release()

    return host_buffer
