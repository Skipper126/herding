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
                                template=_prepare_template(env_data.config._asdict()))

    return module


def _prepare_template(config):
    config.update({
        "dog_color": str(config['colors']['dog']).replace('[', '{').replace(']','}'),
        "sheep_color": str(config['colors']['sheep']).replace('[', '{').replace(']','}'),
        "target_color": str(config['colors']['target']).replace('[', '{').replace(']','}'),
        "barking_dog_color": str(config['colors']['barking_dog']).replace('[', '{').replace(']','}')
    })
    return config
