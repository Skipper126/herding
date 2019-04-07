from herding.data import configuration
from herding.data.env_data import Config, EnvData, Buffers
from herding import opencl
import numpy as np


def create_env_data(params):
    config = _create_config(params)
    ocl = _create_opencl(config)
    shared_buffers = _create_buffers(ocl, config)

    env_data = EnvData(config, shared_buffers, ocl)
    _init_seed(env_data)
    return env_data


def _create_config(params):
    config_dict = configuration.get_default_configuration()
    config_dict.update(params)
    config_internal = configuration.get_internal_configuration()
    config_dict.update(config_internal)
    config = Config(**config_dict)

    return config


def _create_opencl(params):
    definitions = configuration.get_kernel_definitions(params)
    ocl = opencl.OpenCL(definitions)

    return ocl


def _create_buffers(ocl, config):
    buffers_info = configuration.get_shared_buffers_info(config)
    buffers_dict = {}
    for buffer_info in buffers_info:
        name = buffer_info.name
        shape = buffer_info.shape
        dtype = buffer_info.dtype
        buffers_dict[name] = ocl.create_buffer(shape, dtype)

    return Buffers(**buffers_dict)


def _init_seed(env_data):
    if env_data.config.seed is not None:
        np.random.seed(env_data.config.seed)
    seed_buffer = env_data.shared_buffers.seed
    seed_array = seed_buffer.map_write()
    rand_array = np.random.randint(0, np.iinfo(np.uint32).max, seed_array.shape, dtype=np.uint32)
    np.copyto(seed_array, rand_array)
    seed_buffer.unmap()
