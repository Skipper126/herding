from herding.envs.assets.agents.agent import Agent
from herding.envs.assets.configuration.names import ConfigName as cn
from herding.envs.assets.configuration.constants import Coordinate as coo
from . import simple_move, complex_move


class Sheep(Agent):

    def __init__(self, env_data, index):
        super().__init__(env_data, index)

        self.max_movement_speed = env_data.config[cn.MAX_MOVEMENT_SPEED]
        self.pos = env_data.shared_data.sheep_positions[index]
        self.dogs_positions = env_data.shared_data.dogs_positions

    def move(self):
        delta_x = 0
        delta_y = 0
        for dog_pos in self.dogs_positions:
            distance = pow(pow((self.pos[coo.X] - dog_pos[coo.X]), 2) +
                           pow((self.pos[coo.Y] - dog_pos[coo.Y]), 2), 0.5)
            if distance < 200:
                if distance < 50:
                    distance = 50
                delta_x += ((self.pos[coo.X] - dog_pos[coo.X]) / distance) * (200 - distance)
                delta_y += ((self.pos[coo.Y] - dog_pos[coo.Y]) / distance) * (200 - distance)

        if delta_x > 50 or delta_y > 50:
            if delta_x > delta_y:
                delta_y = delta_y / delta_x * 50
                delta_x = 50
            else:
                delta_x = delta_x / delta_y * 50
                delta_y = 50

        delta_x = delta_x / 50 * self.max_movement_speed
        delta_y = delta_y / 50 * self.max_movement_speed
        self.pos[coo.X] += delta_x
        self.pos[coo.Y] += delta_y

