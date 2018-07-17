import math
import numpy as np
from herding.envs.assets.configuration.names import ConfigName as cn
from herding.envs.assets.configuration.constants import Coordinate as coo
from herding.envs.assets.configuration.names import RotationMode
from .ray_casting import RayCasting
from ..agent import Agent
DEG2RAD = 0.01745329252


class Dog(Agent):

    def __init__(self, env_data, index):
        super().__init__(env_data, index)

        self.max_movement_speed = env_data.config[cn.MAX_MOVEMENT_SPEED]
        self.max_rotation_speed = env_data.config[cn.MAX_ROTATION_SPEED]
        self.rotation_mode = env_data.config[cn.ROTATION_MODE]
        self.herd_centre = env_data.shared_data.herd_centre
        self.pos = env_data.shared_data.dogs_positions[index]
        self.rotation = 0

        self.ray_casting = RayCasting(self, env_data)

    def move(self, action):
        delta_x = action[0] * self.max_movement_speed
        delta_y = action[1] * self.max_movement_speed

        vec_length = math.sqrt(delta_x*delta_x + delta_y * delta_y)
        if vec_length > self.max_movement_speed:
            norm = self.max_movement_speed / vec_length
            delta_x *= norm
            delta_y *= norm

        if self.rotation_mode is RotationMode.FREE:
            self.rotation += action[2] * self.max_rotation_speed * DEG2RAD
            self.rotation = self.rotation % (2 * math.pi)
        else:
            self.rotation = np.arctan2(self.pos[coo.Y] - self.herd_centre[coo.Y],
                                       self.pos[coo.X] - self.herd_centre[coo.X]) + 90 * DEG2RAD

        cos_rotation = math.cos(self.rotation)
        sin_rotation = math.sin(self.rotation)
        self.pos[coo.X] += delta_x * cos_rotation + delta_y * sin_rotation
        self.pos[coo.Y] += delta_y * -cos_rotation + delta_x * sin_rotation

    def update_observation(self):
        self.ray_casting.update_observation()
