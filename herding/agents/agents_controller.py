from herding import cuda, data
import numpy as np
import os


class AgentsController:

    def __init__(self, env_data: data.EnvData):
        self.env_data = env_data
        self.thread_count = self._get_thread_count(env_data)
        self.module = self._get_device_module(env_data)
        self.device_move_agents = self.module.get_function('move_agents')
        self.device_cast_rays = self.module.get_function('cast_rays')

    def move_agents(self, action):
        cuda.memcpy_htod(self.env_data.device_action, self._convert_action_input(action))
        self.device_move_agents(self.env_data.device_arrays, block=(self.thread_count, 1, 1))
        cuda.memcpy_dtoh(self.env_data.host_arrays, self.env_data.device_arrays)

    def get_observation(self) -> np.ndarray:
        self.device_cast_rays(self.env_data.device_arrays, block=(self.env_data.config.dogs_count,
                                                                  self.env_data.config.rays_count, 1))
        cuda.memcpy_dtoh(self.env_data.observation, self.env_data.device_observation)

        return self.env_data.observation

    def close(self):
        pass

    @staticmethod
    def _get_device_module(env_data):
        gpu_env_data_header_path = data.get_env_data_header_path()
        kernels_dir = os.path.join(os.path.dirname(__file__), 'kernels')
        module = cuda.compile_files(header_files=[gpu_env_data_header_path,
                                                  os.path.join(kernels_dir, 'declarations.cuh')],
                                    files=[os.path.join(kernels_dir, 'agents_move.cu'),
                                           os.path.join(kernels_dir, 'ray_casting.cu'),
                                           os.path.join(kernels_dir, 'sheep_simple_move.cu'),
                                           os.path.join(kernels_dir, 'dogs_move.cu'),],
                                    template=env_data.config._asdict())

        return module

    @staticmethod
    def _get_thread_count(env_data):
        return max(env_data.config.dogs_count, env_data.config.sheep_count)

    @staticmethod
    def _convert_action_input(action):
        return np.array(list(action), dtype=np.float32)
