import numpy as np
from herding import cuda
from herding import data


class AgentsLayout:

    def __init__(self, env_data: data.EnvData):
        self.env_data = env_data
        self.set_up_function = self._random

    def set_up_agents(self):
        # TODO
        self.set_up_function()

    def _random(self):
        bottom = 50
        top = 600

        self.env_data.dogs_positions[::] = np.random.randint(bottom, top,
                                                            size=(self.env_data.config.dogs_count, 2))
        self.env_data.dogs_rotations[::] = 0
        self.env_data.sheep_positions[::] = np.random.randint(bottom, top,
                                                             size=(self.env_data.config.sheep_count, 2))
        self.env_data.target[::] = np.random.randint(bottom, top, size=(2,1))
        cuda.memcpy_htod(self.env_data.device_arrays, self.env_data.host_arrays)


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
