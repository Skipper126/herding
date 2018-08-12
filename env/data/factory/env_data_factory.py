from env.data.env_data import EnvData
from env.data.factory import host_data, device_data


def create_env_data(params) -> EnvData:
    config = host_data.create_config(params)
    dogs_positions = host_data.create_dogs_positions_array(config)
    sheep_positions = host_data.create_sheep_positions_array(config)
    herd_centre = host_data.create_herd_centre_array()
    observation = host_data.create_observation_array(config)
    dogs_rotations = host_data.create_dogs_rotations_array(config)


    gpu_config = device_data.create_gpu_config(config)
    gpu_dogs_positions = device_data.create_gpu_dogs_positions_array(dogs_positions)
    gpu_sheep_positions = device_data.create_gpu_sheep_positions_array(sheep_positions)
    gpu_herd_centre = device_data.create_gpu_herd_centre_array(herd_centre)
    gpu_observation = device_data.create_gpu_observation_array(observation)
    gpu_dogs_rotations = device_data.create_gpu_dogs_rotations_array(dogs_rotations)
    gpu_action = device_data.create_gpu_action_array(config)

    gpu_env_data = device_data.create_gpu_env_data(
        gpu_config=gpu_config,
        gpu_observation=gpu_observation,
        gpu_dogs_positions=gpu_dogs_positions,
        gpu_sheep_positions=gpu_sheep_positions,
        gpu_dogs_rotations=gpu_dogs_rotations,
        gpu_herd_centre=gpu_herd_centre,
        gpu_action=gpu_action
    )

    env_data = EnvData(
        config=config,
        observation=observation,
        dogs_positions=dogs_positions,
        sheep_positions=sheep_positions,
        dogs_rotations=dogs_rotations,
        herd_centre=herd_centre,
        gpu_env_data=gpu_env_data,
        gpu_config=gpu_config,
        gpu_observation=gpu_observation,
        gpu_dogs_positions=gpu_dogs_positions,
        gpu_sheep_positions=gpu_sheep_positions,
        gpu_dogs_rotations=gpu_dogs_rotations,
        gpu_herd_centre=gpu_herd_centre,
        gpu_action=gpu_action
    )

    return env_data


