from env import cuda
import numpy as np

def create_gpu_env_data(config,
                        observation,
                        dogs_positions,
                        sheep_positions,
                        dogs_rotations,
                        herd_centre):

    gpu_env_data = cuda.malloc(_get_gpu_env_data_size(config))
    gpu_config = cuda.memcpy_htod(gpu_env_data, _get_prepared_gpu_config(config))
    gpu_dogs_positions = cuda.memcpy_htod(gpu_env_data + 11, dogs_positions)
    gpu_sheep_positions = device_data.create_gpu_sheep_positions_array(sheep_positions)
    gpu_herd_centre = device_data.create_gpu_herd_centre_array(herd_centre)
    gpu_observation = device_data.create_gpu_observation_array(observation)
    gpu_dogs_rotations = device_data.create_gpu_dogs_rotations_array(dogs_rotations)
    gpu_action = device_data.create_gpu_action_array(config)
    return 0


def _get_gpu_env_data_size(config):
    return 4* \
           (11 +
           (config.dogs_count * 2) +
           (config.sheep_count * 2) +
            2 +
           (config.dogs_count * config.rays_count) +
           (config.dogs_count))

def _get_prepared_gpu_config(config):
    return np.zeros((11,)).astype(np.float32)

def create_gpu_config(config):
    return 0


def create_gpu_dogs_positions_array(dogs_positions):
    return 0


def create_gpu_dogs_rotations_array(dogs_rotations):
    return 0


def create_gpu_sheep_positions_array(sheep_positions):
    return 0


def create_gpu_observation_array(observation):
    return 0


def create_gpu_herd_centre_array(herd_centre):
    return 0


def create_gpu_action_array(config):
    return 0
