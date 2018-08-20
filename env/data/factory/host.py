from env import configuration
from env.data.factory import info
import numpy as np


def get_host_env_data(params):
    config = _create_config(params)
    env_data_info = info.get_gpu_env_data_info(config)

    env_data = np.empty((env_data_size,))

    return {
        "config": config,
        "dogs_positions": _create_dogs_positions_array(config),
        "sheep_positions": _create_sheep_positions_array(config),
        "herd_centre": _create_herd_centre_array(),
        "observation": _create_observation_array(config),
        "dogs_rotations": _create_dogs_rotations_array(config)
    }


def _create_config(params):
    config = configuration.get_default_configuration()
    config.update(params)

    return config


def _create_dogs_positions_array(config):
    dogs_count = config.dogs_count
    dogs_positions = np.zeros((dogs_count, 2)).astype(np.float32)

    return dogs_positions


def _create_dogs_rotations_array(config):
    dogs_count = config.dogs_count
    dogs_rotations = np.zeros((dogs_count,)).astype(np.float32)

    return dogs_rotations


def _create_sheep_positions_array(config):
    sheep_count = config.sheep_count
    sheep_positions = np.zeros((sheep_count, 2)).astype(np.float32)

    return sheep_positions


def _create_observation_array(config):
    dogs_count = config.dogs_count
    rays_count = config.rays_count
    observation = np.zeros((dogs_count, rays_count)).astype(np.float32)

    return observation


def _create_herd_centre_array():
    herd_centre = np.zeros((2,)).astype(np.float32)

    return herd_centre
