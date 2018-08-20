import numpy as np
from env import cuda
from env import data


class AgentsLayout:

    def __init__(self, env_data):
        self.env_data = env_data
        gpu_env_data_header_path = data.get_env_data_header_path()
        self.module = cuda.compile_files(header_files=[gpu_env_data_header_path],
                                         files=['kernels/random.cu'],
                                         templates=dict(env_data.config))
        self.gpu_random = self.module.get_function('random')

    def set_up_agents(self):
        # TODO
        self.gpu_random(self.env_data.gpu_env_data)
        data.sync.sync_sheep_positions(self.env_data)
        data.sync.sync_dogs_rotations(self.env_data)
        data.sync.sync_dogs_positions(self.env_data)

    def _random(self):
        padding = 5

        for dog_pos in self.dogs_positions:
            x = np.random.randint(self.agent_radius + padding, self.map_width - self.agent_radius - padding)
            y = np.random.randint(self.agent_radius + padding, self.map_height - self.agent_radius - padding)
            dog_pos[coo.X] = x
            dog_pos[coo.Y] = y

        for sheep_pos in self.sheep_positions:
            x = np.random.randint(self.agent_radius + padding, self.map_width - self.agent_radius - padding)
            y = np.random.randint(self.agent_radius + padding, self.map_height - self.agent_radius - padding)
            sheep_pos[coo.X] = x
            sheep_pos[coo.Y] = y
    # TODO
    # @staticmethod
    # def _layout1(env):
    #     sheep_padding = 5
    #     for agent in env.sheep_list:
    #         x = random.randint(agent.radius + sheep_padding, env.map_width - agent.radius - sheep_padding)
    #         y = random.randint(agent.radius + sheep_padding + 200, env.map_height - agent.radius - sheep_padding)
    #         agent.set_pos(x, y)
    #
    #     for i, agent in enumerate(env.dog_list):
    #         x = (i + 1) * (env.map_width / (env.dog_count + 1))
    #         y = 0
    #         agent.set_pos(x, y)
    #
    # @staticmethod
    # def _layout2(env):
    #     # TODO
    #     pass
