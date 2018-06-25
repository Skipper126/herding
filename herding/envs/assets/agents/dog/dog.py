import math
import numpy as np
from .agent import ActiveAgent
from herding.envs.assets.agents.entities import constants
DEG2RAD = 0.01745329252


class Dog(ActiveAgent):

    RAYS = 0
    TARGETS = 1
    LENGTH_TO_CENTER = 0
    TAN_TO_CENTER = 1

    def __init__(self, env):
        super().__init__(env)
        
        self.rotation_mode = env.rotation_mode
        self.ray_count = env.ray_count
        self.ray_length = env.ray_length
        self.max_movement_speed = env.max_movement_speed
        self.max_rotation_speed = env.max_rotation_speed
        self.field_of_view = env.field_of_view
        self.herd_centre_point = env.herd_centre_point
        
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

    def clear_observation(self):
        for i, _ in enumerate(self.observation[self.RAYS]):
            self.observation[self.RAYS][i] = 0
            self.observation[self.TARGETS][i] = 0

    def get_distance_from_agent(self, agent):
        return pow(pow((self.x - agent.x), 2) + pow((self.y - agent.y), 2), 0.5)

    def calculate_angle(self, agent):
        temp_angle = math.atan2(self.y - agent.y, self.x - agent.x) - self.rotation
        while temp_angle < 0:
            temp_angle += 2 * math.pi
        return temp_angle

    def calculate_delta(self, rayTan, agent):
        return pow((2 * (self.x - agent.x)) + (2 * rayTan * (self.y - agent.y)), 2) - (4 * (1 + pow(rayTan, 2)) * (-1 * pow(self.radius, 2) + pow(self.x - agent.x, 2) + pow(self.y - agent.y, 2)))

    def calculate_straight_to_circle_distance(self, agent, index):
        return abs(-1 * math.tan(self.rotation - self.ray_radian[index]) * (self.x - agent.x) + self.y - agent.y) / pow(pow(math.tan(self.rotation - self.ray_radian[index]), 2) + 1, 0.5)

    def is_in_sight(self, tempAngle):
        if self.wide_view:
            if not self.ray_radian[self.ray_count-1] < tempAngle < self.ray_radian[0]:
                return True
        else:
            if self.ray_radian[0] < tempAngle < self.ray_radian[self.ray_count-1]:
                return True
        return False

    def set_distance_and_color(self, index, agent):
        ray_tan = math.tan(self.rotation - self.ray_radian[index])
        delta = self.calculate_delta(ray_tan, agent)
        x1 = (((2 * (self.x - agent.x)) + (2 * ray_tan * (self.y - agent.y))) - math.pow(delta, 0.5)) / (2 * (1 + pow(ray_tan, 2)))
        y1 = ray_tan * x1
        x2 = (((2 * (self.x - agent.x)) + (2 * ray_tan * (self.y - agent.y))) + math.pow(delta, 0.5)) / (2 * (1 + pow(ray_tan, 2)))
        y2 = ray_tan * x2
        distance1 = pow(pow(x1, 2) + pow(y1, 2), 0.5)
        distance2 = pow(pow(x2, 2) + pow(y2, 2), 0.5)
        if distance1 < distance2:
            distance = distance1 - self.radius
        else:
            distance = distance2 - self.radius
        if 1 - (distance / self.ray_length) > self.observation[self.RAYS][index]:
            self.observation[self.RAYS][index] = 1 - (distance / self.ray_length)
            self.observation[self.TARGETS][index] = 1 if type(agent) is Dog else -1

    def iterate_rays(self, distance, agent, index, iterator):
        while 0 <= index <= self.ray_count - 1:
            circle_distance = self.calculate_straight_to_circle_distance(agent, index)
            if circle_distance <= self.radius:
                if (distance - (2 * self.radius)) / self.ray_length < 1 - self.observation[self.RAYS][index]:
                    self.set_distance_and_color(index, agent)
            else:
                break
            index += iterator

    def color_rays(self, tempAngle, distance, agent):
        if tempAngle < self.ray_radian[0]:
            tempAngle += 2 * math.pi
        left = self.ray_count - 2 - int((tempAngle - self.ray_radian[0]) / ((self.field_of_view / (self.ray_count - 1)) * DEG2RAD))
        right = left + 1
        # color left rays
        self.iterate_rays(distance, agent, left, -1)
        # color right rays
        self.iterate_rays(distance, agent, right, 1)

    def update_observation_to_center(self):
        last_index = self.ray_count
        abs_x = abs(self.x - self.herd_centre_point[0])
        abs_y = abs(self.y - self.herd_centre_point[1])
        self.observation[self.LENGTH_TO_CENTER][last_index] = pow(pow(abs_x, 2) + pow(abs_y, 2), 0.5) / self.ray_length
        self.observation[self.TAN_TO_CENTER][last_index] = (((np.arctan2(abs_x, abs_y) + self.rotation) % (2 * math.pi)) * 2) / (2 * math.pi) - 1

    def get_observation(self):
        self.clear_observation()
        for agent in self.sheep_list + self.dog_list:
            if agent is self:
                continue
            distance = self.get_distance_from_agent(agent)
            if distance - (2 * self.radius) < self.ray_length:
                temp_angle = self.calculate_angle(agent)
                if self.is_in_sight(temp_angle):
                    self.color_rays(temp_angle, distance, agent)
        #self.update_observation_to_center()

        return self.observation
