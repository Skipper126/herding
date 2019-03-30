from herding.data import EnvData
import numpy as np


class AgentsController:

    def __init__(self, env_data: EnvData):
        self.dogs_count = env_data.config.dogs_count
        self.rays_count = env_data.config.rays_count
        self.sheep_count = env_data.config.sheep_count

        self.action_buffer = env_data.buffers.action
        self.observation_buffer = env_data.buffers.observation

        self.agents_move_kernel = env_data.ocl.create_module('herding/agents/agents_move.cl',
                                                             'move_agents',
                                                             [env_data.buffers.dogs_positions,
                                                              env_data.buffers.dogs_rotations,
                                                              env_data.buffers.sheep_positions])
        self.observation_kernel = env_data.ocl.create_module(env_data,
                                                             'herding/agents/observation.cl',
                                                             'get_observation',
                                                             [])

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

    @staticmethod
    def _sanitize_action_input(action):
        if type(action) is np.ndarray:
            return action
        else:
            return np.array(list(action), dtype=np.float32)
