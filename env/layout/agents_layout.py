from openaigym.envs.assets.configuration.names import ConfigName as cn
from openaigym.envs.assets.configuration.constants import Coordinate as coo
import numpy as np


class AgentsLayout:

    def __init__(self, env_data):
        self.dogs_positions = env_data.shared_data.dogs_positions
        self.sheep_positions = env_data.shared_data.sheep_positions
        self.dogs_rotations = env_data.shared_data.dogs_rotations
        self.agent_radius = env_data.config[cn.AGENT_RADIUS]
        self.map_width = env_data.config[cn.MAP_WIDTH]
        self.map_height = env_data.config[cn.MAP_HEIGHT]

    def set_up_agents(self):
        # TODO
        self._random()

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
