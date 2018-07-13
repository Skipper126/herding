import math
import numpy as np
from .agent import ActiveAgent
from herding.envs.assets import constants
from .ray_casting import RayCasting
DEG2RAD = 0.01745329252


class Dog:

    RAYS = 0
    TARGETS = 1
    LENGTH_TO_CENTER = 0
    TAN_TO_CENTER = 1

    def __init__(self, params, shared_data, index):
        
        self.rotation_mode = params.rotation_mode
        self.ray_count = params.ray_count
        self.ray_length = params.ray_length
        self.max_movement_speed = params.max_movement_speed
        self.max_rotation_speed = params.max_rotation_speed
        self.field_of_view = params.field_of_view
        self.herd_centre_point = params.herd_centre_point
        self.ray_casting = RayCasting(params)
        
        self.rotation = 0
        self.ray_radian = []
        
        for i in range(self.ray_count):
            self.ray_radian.append((math.pi + ((180 - self.field_of_view) / 360) * math.pi + (self.field_of_view / (self.ray_count - 1)) * DEG2RAD * i) % (2 * math.pi))
        if self.ray_radian[0] > self.ray_radian[self.ray_count - 1]:
            self.wide_view = True
        else:
            self.wide_view = False

        for i, _ in enumerate(self.observation[self.RAYS]):
            self.observation[self.RAYS][i] = 0
            self.observation[self.TARGETS][i] = 0

    def move(self, action):
        delta_x = action[0] * self.max_movement_speed
        delta_y = action[1] * self.max_movement_speed

        vec_length = math.sqrt(delta_x*delta_x + delta_y * delta_y)
        if vec_length > self.max_movement_speed:
            norm = self.max_movement_speed / vec_length
            delta_x *= norm
            delta_y *= norm

        if self.rotation_mode is constants.RotationMode.FREE:
            self.rotation += action[2] * self.max_rotation_speed * DEG2RAD
            self.rotation = self.rotation % (2 * math.pi)
        else:
            self.rotation = np.arctan2(self.y - self.herd_centre_point[1], self.x - self.herd_centre_point[0]) + 90 * DEG2RAD

        cos_rotation = math.cos(self.rotation)
        sin_rotation = math.sin(self.rotation)
        self.x += delta_x * cos_rotation + delta_y * sin_rotation
        self.y += delta_y * -cos_rotation + delta_x * sin_rotation

    def update_observation(self):
        pass
