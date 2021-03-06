import tests.test_env_step
from herding.agents.matrix_sorter import MatrixSorter
from herding.data import EnvData
import numpy as np


class AgentsController:

    def __init__(self, env_data: EnvData):
        self.dogs_count = env_data.config.dogs_count
        self.rays_count = env_data.config.rays_count
        self.sheep_count = env_data.config.sheep_count
        self.matrix_side_length = env_data.config.agents_matrix_side_length
        self.n_matrix_side_length = env_data.config.neighbours_matrix_side_length
        self.env_data = env_data
        self.matrix_sorter = MatrixSorter(env_data)
        self.action_buffer = env_data.ocl.create_buffer((self.dogs_count, 2), np.int32)
        self.observation_buffer = env_data.shared_buffers.observation

        # Observation is by default mapped to host memory.
        # It is only unmapped in get_observation and mapped again.
        self.observation_buffer.map_read()
        self.env_data.shared_buffers.reward.map_read()

    def step(self, action):
        action_map = self.action_buffer.map_write()
        np.copyto(action_map, action.astype(np.int32))
        self.action_buffer.unmap()
        self.observation_buffer.unmap()

        env_step_kernel = self.env_data.ocl.create_module('herding/agents/agents_move.cl',
                                                     'env_step',
                                                     [self.env_data.shared_buffers.input_matrix,
                                                      self.env_data.shared_buffers.output_matrix,
                                                      self.action_buffer,
                                                      self.env_data.shared_buffers.observation])

        env_step_kernel.run((self.matrix_side_length, self.matrix_side_length, self.n_matrix_side_length ** 2),
                                 (1, 1, self.n_matrix_side_length ** 2))

        self.env_data.shared_buffers.output_matrix, self.env_data.shared_buffers.input_matrix = \
            self.env_data.shared_buffers.input_matrix, self.env_data.shared_buffers.output_matrix

        self.matrix_sorter.sort_single_pass()

        return self.observation_buffer.map_read(), 0



    def step_cpu(self, action):
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
