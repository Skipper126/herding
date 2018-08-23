from env import cuda, data
import numpy as np


class AgentsController:

    def __init__(self, env_data: data.EnvData):
        self.env_data = env_data
        self.thread_count = self._get_thread_count(env_data)
        self.module = self._get_device_module(env_data)
        self.device_move_agents = self.module.get_function('move_agents')
        self.device_cast_rays = self.module.get_function('cast_rays')

    def move_agents(self, action):
        cuda.memcpy_htod(self.env_data.device_action, action)
        self.device_move_agents(self.env_data.device_arrays, block=(self.thread_count, 1, 1))
        cuda.memcpy_dtoh(self.env_data.host_arrays, self.env_data.device_arrays)

    def get_observation(self) -> np.ndarray:
        self.device_cast_rays()
        cuda.memcpy_dtoh(self.env_data.observation, self.env_data.device_observation)

        return self.env_data.observation

    def close(self):
        pass

    @staticmethod
    def _get_device_module(env_data):
        gpu_env_data_header_path = data.get_env_data_header_path()
        module = cuda.compile_files(header_files=[gpu_env_data_header_path,
                                                  'kernels/declarations.cuh'],
                                    files=['kernels/agents_move.cu',
                                           'kernels/ray_casting.cu'],
                                    templates=dict(env_data.config))

        return module

    @staticmethod
    def _get_thread_count(env_data):
        return max(env_data.config.dogs_count, env_data.config.sheep_count)
