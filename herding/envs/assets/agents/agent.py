import copy
import numpy as np


class Agent:

    def __init__(self, env):
        self.x = 0
        self.y = 0
        self.radius = env.agent_radius
        self.dog_list = None
        self.sheep_list = None

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def set_lists(self, dog_list, sheep_list):
        self.dog_list = dog_list
        self.sheep_list = sheep_list


class PassiveAgent(Agent):

    def move(self):
        raise NotImplementedError


class ActiveAgent(Agent):

    def __init__(self, env):
        super().__init__(env)

    def move(self, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError
