

def get_gpu_env_data_info(config):
    return {
        'dogs_positions': config.dogs_count * 2 * 4,
        'sheep_positions': config.sheep_count * 2 * 4,
        'herd_centre': 2 * 4,
        'observation': config.dogs_count * config.rays_count * 4,
        'dogs_rotations': config.dogs_count * 4,
        'action': config.dogs_count * 3 * 4
    }


def get_gpu_env_data_size(gpu_env_data_info):
    size = 0
    for key, value in gpu_env_data_info.items():
        size += value

    return size


def get_gpu_env_data_offsets(gpu_env_data_info):
    offset = 0
    info = {}

    for key, value in gpu_env_data_info.items():
        info[key] = offset
        offset += value

    return info