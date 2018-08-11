import os
import json
from env.data.env_data import EnvData
from env.data.gpu_env_data_factory import create_gpu_env_data
from env.data.factory import host_data, device_data


def create_env_data(params) -> EnvData:
    config = host_data.create_config(params)
    observation = host_data.create_observation_array(config)
    dogs_positions = host_data.create_dogs_positions_array(config)
    sheep_positions = host_data.create_sheep_positions_array(config)
    dogs_rotations = host_data.create_dogs_rotations_array(config)
    herd_centre = host_data.create_herd_centre_array(config)

    env_data = EnvData(
        config=config,
        observation=observation,
        dogs_positions=dogs_positions,
        sheep_positions=sheep_positions,
        dogs_rotations=dogs_rotations,
        herd_centre=herd_centre,
        gpu_config=device_data.create_gpu_config(config),
        gpu_observation=device_data.create_gpu_observation_array(observation),
        gpu_dogs_positions=device_data.create_gpu_dogs_positions_array(dogs_positions),
        gpu_sheep_positions=device_data.create_gpu_sheep_positions_array(sheep_positions),
        gpu_dogs_rotations=device_data.create_gpu_dogs_rotations_array(dogs_rotations),
        gpu_herd_centre=device_data.create_gpu_herd_centre_array(herd_centre),
        gpu_action=device_data.create_gpu_action_array(config)
    )

    return env_data


