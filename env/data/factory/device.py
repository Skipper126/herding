from env import cuda


def get_gpu_env_data(host_env_data):
    gpu_env_data_info = _get_gpu_env_data_info(host_env_data['config'])
    gpu_env_data_size = _get_gpu_env_data_size(gpu_env_data_info)
    gpu_env_data_offsets = _get_gpu_env_data_offsets(gpu_env_data_info)
    gpu_env_data = cuda.malloc(gpu_env_data_size)

    return {
        'gpu_env_data': gpu_env_data,
        'gpu_dogs_positions': gpu_env_data + gpu_env_data_offsets['gpu_dogs_positions'],
        'gpu_sheep_positions': gpu_env_data + gpu_env_data_offsets['gpu_sheep_positions'],
        'gpu_herd_centre': gpu_env_data + gpu_env_data_offsets['gpu_herd_centre'],
        'gpu_observation': gpu_env_data + gpu_env_data_offsets['gpu_observation'],
        'gpu_dogs_rotations':  gpu_env_data + gpu_env_data_offsets['gpu_dogs_rotations'],
        'gpu_action':  gpu_env_data + gpu_env_data_offsets['gpu_action']
    }



