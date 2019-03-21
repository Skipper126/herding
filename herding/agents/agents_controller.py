from herding import opencl, data
import numpy as np


class AgentsController:

    def __init__(self, env_data: data.EnvData):
        self.dogs_count = env_data.config.dogs_count
        self.rays_count = env_data.config.rays_count
        self.sheep_count = env_data.config.sheep_count

        self.action_mapping = opencl.create_buffer_mapping(env_data, 'action')
        self.observation_mapping = opencl.create_buffer_mapping(env_data, 'observation')

        self.agents_move_kernel = opencl.create_module(env_data.ocl,
                                                       'herding/agents/agents_move.cl',
                                                       'move_agents')
        self.observation_kernel = opencl.create_module(env_data.ocl,
                                                       'herding/agents/observation.cl',
                                                       'get_observation')

        self.agents_move_workers = max(self.dogs_count, self.sheep_count)

        # Observation is by default mapped to host memory.
        # It is only unmapped in get_observation and mapped again.
        self.observation_mapping.map_read()

    def move_agents(self, action):
        action_map = self.action_mapping.map_write()
        input_action = self._sanitize_action_input(action)
        np.copyto(action_map, input_action)
        self.action_mapping.unmap()
        self.agents_move_kernel.run(self.agents_move_workers)

    def get_observation(self) -> np.ndarray:
        self.observation_mapping.unmap()
        self.observation_kernel.run(self.dogs_count * self.rays_count)
        observation = self.observation_mapping.map_read()

        return observation

    def close(self):
        pass

    @staticmethod
    def _sanitize_action_input(action):
        if type(action) is np.ndarray:
            return action
        else:
            return np.array(list(action), dtype=np.float32)
