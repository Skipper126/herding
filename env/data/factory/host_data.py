from env import configuration
import numpy as np


def create_config(params):
    config = configuration.get_default_configuration()
    config.update(params)

    return config


def create_dogs_positions_array(config):
    dogs_count = config.dogs_count
    dogs_positions = np.zeros((dogs_count, 2)).astype(np.float32)

    return dogs_positions


def create_dogs_rotations_array(config):
    dogs_count = config.dogs_count
    dogs_rotations = np.zeros((dogs_count,)).astype(np.float32)

    return dogs_rotations


def create_sheep_positions_array(config):
    sheep_count = config.sheep_count
    sheep_positions = np.zeros((sheep_count, 2)).astype(np.float32)

    return sheep_positions


def create_observation_array(config):
    dogs_count = config.dogs_count
    rays_count = config.rays_count
    observation = np.zeros((dogs_count, rays_count)).astype(np.float32)

    return observation


def create_herd_centre_array():
    herd_centre = np.zeros((2,)).astype(np.float32)

    return herd_centre