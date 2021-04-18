from herding.data import configuration
from herding.data.env_data import Config, EnvData, Buffers
from herding import opencl
import numpy as np


def create_config(params) -> Config:
    config_dict = configuration.get_default_configuration()
    config_internal = configuration.get_internal_configuration()
    config_dict.update(config_internal)
    if params is not None:
        config_dict.update(params)

    return Config(**config_dict)


def init_opencl(env_data: EnvData):
    definitions = configuration.get_kernel_definitions(env_data.config)
    env_data.ocl = opencl.OpenCL(env_data, definitions)
    env_data.shared_buffers = _create_buffers(env_data.ocl, env_data.config)
    _init_seed(env_data)


def _create_buffers(ocl: opencl.OpenCL, config: Config) -> Buffers:
    buffers_info = configuration.get_shared_buffers_info(config)
    buffers_dict = {}
    for buffer_info in buffers_info:
        name = buffer_info.name
        shape = buffer_info.shape
        dtype = buffer_info.dtype
        buffers_dict[name] = ocl.create_buffer(shape, dtype)

    buffers_dict['current_agents_matrix'] = buffers_dict['agents_matrix1']

    return Buffers(**buffers_dict)


def _init_seed(env_data: EnvData):
    if env_data.config.seed is not None:
        np.random.seed(env_data.config.seed)
    seed_buffer = env_data.shared_buffers.seed
    seed_array = seed_buffer.map_write()
    rand_array = np.random.randint(0, np.iinfo(np.uint32).max, seed_array.shape, dtype=np.uint32)
    np.copyto(seed_array, rand_array)
    seed_buffer.unmap()
