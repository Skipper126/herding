from herding import cuda, data
from herding.agents.factory import get_device_module
import numpy as np


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

        self.module = get_device_module(env_data)
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
    def _get_thread_count(env_data):
        return max(env_data.config.dogs_count, env_data.config.sheep_count)

    @staticmethod
    def _convert_action_input(action):
        if type(action) is np.ndarray:
            return action
        else:
            return np.array(list(action), dtype=np.float32)


