from herding.data import EnvData
import numpy as np


class AgentsController:

    def __init__(self, env_data: EnvData):
        self.dogs_count = env_data.config.dogs_count
        self.rays_count = env_data.config.rays_count
        self.sheep_count = env_data.config.sheep_count

        self.action_buffer = env_data.ocl.create_buffer((self.dogs_count, 3), np.float32)
        self.observation_buffer = env_data.shared_buffers.observation

        self.move_dogs_kernel = env_data.ocl.create_module('herding/agents/agents_move.cl',
                                                           'move_dogs',
                                                           [env_data.shared_buffers.dogs_positions,
                                                            self.action_buffer])
        self.move_sheep_kernel = env_data.ocl.create_module('herding/agents/agents_move.cl',
                                                            'move_sheep_simple',
                                                            [env_data.shared_buffers.dogs_positions,
                                                             env_data.shared_buffers.sheep_positions])
        self.observation_kernel = env_data.ocl.create_module('herding/agents/observation.cl',
                                                             'get_observation',
                                                             [env_data.shared_buffers.dogs_positions,
                                                              env_data.shared_buffers.sheep_positions,
                                                              env_data.shared_buffers.target_position,
                                                              env_data.shared_buffers.observation,
                                                              env_data.shared_buffers.rays_lengths])

        # Observation is by default mapped to host memory.
        # It is only unmapped in get_observation and mapped again.
        self.observation_buffer.map_read()

    def move_agents(self, action):
        action_map = self.action_buffer.map_write()
        input_action = self._sanitize_action_input(action)
        np.copyto(action_map, input_action)
        self.action_buffer.unmap()
        self.move_dogs_kernel.run((self.dogs_count,), (1,))
        self.move_sheep_kernel.run((self.sheep_count,), (1,))

    def get_observation(self) -> np.ndarray:
        self.observation_buffer.unmap()
        self.observation_kernel.run((self.dogs_count, self.rays_count), None)
        observation = self.observation_buffer.map_read()

        return observation

    @staticmethod
    def _sanitize_action_input(action):
        if type(action) is np.ndarray:
            return action
        else:
            return np.array(list(action), dtype=np.float32)
