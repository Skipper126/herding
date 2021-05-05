import tests.test_env_step
from herding.agents.matrix_sorter import MatrixSorter
from herding.data import EnvData
import numpy as np


class AgentsController:

    def __init__(self, env_data: EnvData):
        self.dogs_count = env_data.config.dogs_count
        self.rays_count = env_data.config.rays_count
        self.sheep_count = env_data.config.sheep_count
        self.env_data = env_data
        self.matrix_sorter = MatrixSorter(env_data)
        self.action_buffer = env_data.ocl.create_buffer((self.dogs_count, 2), np.int32)
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
        self.env_data.shared_buffers.reward.map_read()

    def step(self, action):
        input_matrix = self.env_data.shared_buffers.input_matrix.map_read()
        output_matrix = self.env_data.shared_buffers.output_matrix.map_write()

        self.observation_buffer.unmap()
        self.env_data.shared_buffers.reward.unmap()
        observations = self.observation_buffer.map_write()
        reward = self.env_data.shared_buffers.reward.map_write()

        action_map = self.action_buffer.map_write()
        np.copyto(action_map, action.astype(np.int32))

        tests.test_env_step.step(input_matrix, output_matrix, action_map, observations, reward, self.env_data)

        self.action_buffer.unmap()

        self.env_data.shared_buffers.input_matrix.unmap()
        self.env_data.shared_buffers.output_matrix.unmap()

        self.env_data.shared_buffers.output_matrix, self.env_data.shared_buffers.input_matrix = \
            self.env_data.shared_buffers.input_matrix, self.env_data.shared_buffers.output_matrix

        self.observation_buffer.unmap()
        self.env_data.shared_buffers.reward.unmap()
        observation = self.observation_buffer.map_read()
        reward = self.env_data.shared_buffers.reward.map_read()

        self.matrix_sorter.sort_single_pass()

        return observation, reward

    def move_agents(self, action):
        action_map = self.action_buffer.map_write()
        np.copyto(action_map, action.astype(np.int32))
        self.action_buffer.unmap()
        self.move_dogs_kernel.run((self.dogs_count,))
        self.move_sheep_kernel.run((self.sheep_count,))

    def get_observation(self) -> np.ndarray:
        self.observation_buffer.unmap()
        self.observation_kernel.run((self.dogs_count, self.rays_count), None)
        observation = self.observation_buffer.map_read()

        return observation
