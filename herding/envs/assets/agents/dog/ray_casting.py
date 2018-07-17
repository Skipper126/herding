import math
import numpy as np
from herding.envs.assets.configuration.names import ConfigName as cn
from herding.envs.assets.configuration.constants import Coordinate as coo
DEG2RAD = 0.01745329252


class RayCasting:
    RAYS = 0
    TARGETS = 1
    LENGTH_TO_CENTER = 0
    TAN_TO_CENTER = 1

    def __init__(self, dog, env_data):
        self.dog = dog
        self.ray_count = env_data.config[cn.RAYS_COUNT]
        self.ray_length = env_data.config[cn.RAY_LENGTH]
        self.field_of_view = env_data.config[cn.FIELD_OF_VIEW]
        self.ray_radian = []
        self.observation = env_data.shared_data.observation[dog.index]
        self.herd_centre = env_data.shared_data.herd_centre

        for i in range(self.ray_count):
            self.ray_radian.append((math.pi + ((180 - self.field_of_view) / 360) * math.pi + (
                        self.field_of_view / (self.ray_count - 1)) * DEG2RAD * i) % (2 * math.pi))
        if self.ray_radian[0] > self.ray_radian[self.ray_count - 1]:
            self.wide_view = True
        else:
            self.wide_view = False

        for i, _ in enumerate(self.observation[self.RAYS]):
            self.observation[self.RAYS][i] = 0
            self.observation[self.TARGETS][i] = 0

    def _clear_observation(self):
        self.observation[:] = 0

    # def _get_distance_from_agent(self, agent):
    #     return pow(pow((self.dog.x - agent.x), 2) + pow((self.dog.y - agent.y), 2), 0.5)
    #
    # def _calculate_angle(self, agent):
    #     temp_angle = math.atan2(self.dog.y - agent.y, self.dog.x - agent.x) - self.dog.rotation
    #     while temp_angle < 0:
    #         temp_angle += 2 * math.pi
    #     return temp_angle
    #
    # def _calculate_delta(self, rayTan, agent):
    #     return pow((2 * (self.dog.x - agent.x)) + (2 * rayTan * (self.dog.y - agent.y)), 2) - (4 * (1 + pow(rayTan, 2)) * (
    #                 -1 * pow(self.dog.radius, 2) + pow(self.dog.x - agent.x, 2) + pow(self.dog.y - agent.y, 2)))
    #
    # def _calculate_straight_to_circle_distance(self, agent, index):
    #     return abs(-1 * math.tan(self.dog.rotation - self.ray_radian[index]) * (self.dog.x - agent.x) + self.dog.y - agent.y) / pow(
    #         pow(math.tan(self.dog.rotation - self.ray_radian[index]), 2) + 1, 0.5)
    #
    # def _is_in_sight(self, tempAngle):
    #     if self.wide_view:
    #         if not self.ray_radian[self.ray_count - 1] < tempAngle < self.ray_radian[0]:
    #             return True
    #     else:
    #         if self.ray_radian[0] < tempAngle < self.ray_radian[self.ray_count - 1]:
    #             return True
    #     return False
    #
    # def _set_distance_and_color(self, index, agent):
    #     ray_tan = math.tan(self.dog.rotation - self.ray_radian[index])
    #     delta = self._calculate_delta(ray_tan, agent)
    #     x1 = (((2 * (self.dog.x - agent.x)) + (2 * ray_tan * (self.dog.y - agent.y))) - math.pow(delta, 0.5)) / (
    #                 2 * (1 + pow(ray_tan, 2)))
    #     y1 = ray_tan * x1
    #     x2 = (((2 * (self.dog.x - agent.x)) + (2 * ray_tan * (self.dog.y - agent.y))) + math.pow(delta, 0.5)) / (
    #                 2 * (1 + pow(ray_tan, 2)))
    #     y2 = ray_tan * x2
    #     distance1 = pow(pow(x1, 2) + pow(y1, 2), 0.5)
    #     distance2 = pow(pow(x2, 2) + pow(y2, 2), 0.5)
    #     if distance1 < distance2:
    #         distance = distance1 - self.dog.radius
    #     else:
    #         distance = distance2 - self.dog.radius
    #     if 1 - (distance / self.ray_length) > self.observation[self.RAYS][index]:
    #         self.observation[self.RAYS][index] = 1 - (distance / self.ray_length)
    #         self.observation[self.TARGETS][index] = 1 if type(agent) is Dog else -1
    #
    # def _iterate_rays(self, distance, agent, index, iterator):
    #     while 0 <= index <= self.ray_count - 1:
    #         circle_distance = self._calculate_straight_to_circle_distance(agent, index)
    #         if circle_distance <= self.dog.radius:
    #             if (distance - (2 * self.dog.radius)) / self.ray_length < 1 - self.observation[self.RAYS][index]:
    #                 self._set_distance_and_color(index, agent)
    #         else:
    #             break
    #         index += iterator
    #
    # def _color_rays(self, tempAngle, distance, agent):
    #     if tempAngle < self.ray_radian[0]:
    #         tempAngle += 2 * math.pi
    #     left = self.ray_count - 2 - int(
    #         (tempAngle - self.ray_radian[0]) / ((self.field_of_view / (self.ray_count - 1)) * DEG2RAD))
    #     right = left + 1
    #     # color left rays
    #     self._iterate_rays(distance, agent, left, -1)
    #     # color right rays
    #     self._iterate_rays(distance, agent, right, 1)
    #
    # def _update_observation_to_center(self):
    #     last_index = self.ray_count
    #     abs_x = abs(self.dog.x - self.herd_centre[0])
    #     abs_y = abs(self.dog.y - self.herd_centre[1])
    #     self.observation[self.LENGTH_TO_CENTER][last_index] = pow(pow(abs_x, 2) + pow(abs_y, 2), 0.5) / self.ray_length
    #     self.observation[self.TAN_TO_CENTER][last_index] = (((np.arctan2(abs_x, abs_y) + self.dog.rotation) % (
    #                 2 * math.pi)) * 2) / (2 * math.pi) - 1

    def update_observation(self):
        self._clear_observation()


        # for agent in self.sheep_list + self.dog_list:
        #     if agent is self:
        #         continue
        #     distance = self._get_distance_from_agent(agent)
        #     if distance - (2 * self.dog.radius) < self.ray_length:
        #         temp_angle = self._calculate_angle(agent)
        #         if self._is_in_sight(temp_angle):
        #             self._color_rays(temp_angle, distance, agent)
        # # self.update_observation_to_center()

