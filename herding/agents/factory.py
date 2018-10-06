from herding import cuda, data
import os


def get_device_module(env_data):
    gpu_env_data_header_path = data.get_env_data_header_path()
    kernels_dir = os.path.join(os.path.dirname(__file__), 'kernels')
    module = cuda.compile_files(header_files=[gpu_env_data_header_path,
                                              os.path.join(kernels_dir, 'declarations.cuh')],
                                files=[os.path.join(kernels_dir, 'agents_move.cu'),
                                       os.path.join(kernels_dir, 'observation.cu'),
                                       os.path.join(kernels_dir, 'sheep_simple_move.cu'),
                                       os.path.join(kernels_dir, 'sheep_complex_move.cu'),
                                       os.path.join(kernels_dir, 'dogs_move.cu')],
                                template=env_data.config._asdict())

    return module

def get_input_memory_buffer(env_data):
    arrays = ['action']
    if env_data.config.sheep_type == 'complex':
        arrays.append('rand_values')

    return data.get_memory_buffer(env_data, arrays)

def get_observation_memory_buffer(env_data):
    return data.get_memory_buffer(env_data, ['observation'])


def get_agents_move_thread_count(env_data):
    return max(env_data.config.dogs_count, env_data.config.sheep_count)
