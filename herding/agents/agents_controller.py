from herding import cuda, data
import numpy as np
import os


class AgentsController:

    def __init__(self, env_data: data.EnvData):
        self.dogs_count = env_data.config.dogs_count
        self.rays_count = env_data.config.rays_count
        self.sheep_count = env_data.config.sheep_count
        self.thread_count = self._get_thread_count(env_data)

        self.host_arrays = env_data.host_arrays
        self.observation = env_data.observation
        self.rand_values = env_data.rand_values

        self.device_arrays = env_data.device_arrays
        self.device_observation = env_data.device_observation
        self.device_rand_values = env_data.device_rand_values
        self.device_action = env_data.device_action

        self.module = self._get_device_module(env_data)
        self.device_move_agents = self.module.get_function('move_agents')
        self.device_get_observation = self.module.get_function('get_observation')

    def move_agents(self, action):
        cuda.memcpy_htod(self.device_action, self._convert_action_input(action))
        self._fill_rand_values()
        self.device_move_agents(self.device_arrays, block=(self.thread_count, 1, 1))
        cuda.memcpy_dtoh(self.host_arrays, self.device_arrays)

    def get_observation(self) -> np.ndarray:
        self.device_get_observation(self.device_arrays, block=(self.dogs_count, self.rays_count, 1))
        cuda.memcpy_dtoh(self.host_arrays, self.device_arrays)

        return self.observation

    def close(self):
        pass

    def _fill_rand_values(self):
        self.rand_values[:] = np.random.rand(self.sheep_count, 1)
        cuda.memcpy_htod(self.device_rand_values, self.rand_values)

    @staticmethod
    def _get_device_module(env_data):
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

    @staticmethod
    def _get_thread_count(env_data):
        return max(env_data.config.dogs_count, env_data.config.sheep_count)

    @staticmethod
    def _convert_action_input(action):
        return np.array(list(action), dtype=np.float32)
