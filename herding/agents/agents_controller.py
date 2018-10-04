from herding import cuda, data
from herding.agents import factory
import numpy as np


class AgentsController:

    def __init__(self, env_data: data.EnvData):
        self.dogs_count = env_data.config.dogs_count
        self.rays_count = env_data.config.rays_count
        self.sheep_count = env_data.config.sheep_count
        self.agents_move_thread_count = factory.get_agents_move_thread_count(env_data)

        self.input_buffer: data.MemoryBuffer = factory.get_input_memory_buffer(env_data)
        self.observation_buffer: data.MemoryBuffer = factory.get_observation_memory_buffer(env_data)
        self.observation = env_data.observation
        self.rand_values = env_data.rand_values
        self.action = env_data.action

        self.device_buffer = env_data.device_buffer
        self.module = factory.get_device_module(env_data)
        self.device_move_agents = self.module.get_function('move_agents')
        self.device_get_observation = self.module.get_function('get_observation')

    def act(self, action) -> np.ndarray:
        self._convert_action_input(action)
        #self._fill_rand_values() TODO
        self.input_buffer.sync_htod()
        self.device_move_agents(self.device_buffer, block=(self.agents_move_thread_count, 1, 1))
        self.device_get_observation(self.device_buffer, block=(self.dogs_count, self.rays_count, 1))
        self.observation_buffer.sync_dtoh()

        return self.observation

    def get_observation(self) -> np.ndarray:
        self.device_get_observation(self.device_buffer, block=(self.dogs_count, self.rays_count, 1))
        self.observation_buffer.sync_dtoh()

        return self.observation

    def close(self):
        pass

    def _fill_rand_values(self):
        np.copyto(self.rand_values, np.random.rand(*self.rand_values.shape))

    def _convert_action_input(self, action):
        converted_action = None
        if type(action) is np.ndarray:
            converted_action = action
        else:
            converted_action = np.array(list(action), dtype=np.float32)
        np.copyto(self.action, converted_action)


