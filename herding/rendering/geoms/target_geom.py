from .geom import *
from gym.envs.classic_control import rendering


class Target(Geom):
    BODY = 0

    def __init__(self):
        self.body = Part(rendering.make_circle(4, res=4))
        self.body.set_color(0, 0, 1)

    def get_parts(self):
        return [self.body.body]

    def update(self, arrays):
        self.body.set_pos(arrays.target_position[0], arrays.target_position[1])
