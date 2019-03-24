from herding.data import configuration
from herding.data.env_data import Config, ArrayInfo, EnvData
from herding import opencl
import numpy as np


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
