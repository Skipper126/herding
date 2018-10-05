from .geom import *
from gym.envs.classic_control import rendering
from herding.data import get_color_tuple_from_config


class Target(Geom):
    BODY = 0

    def __init__(self, env_data):
        self.target_pos = env_data.target
        self.body = Part(rendering.make_circle(4, res=4))
        self.body.set_color(0, 0, 1)

    def get_parts(self):
        return [self.body.body]

    def update(self):
        self.body.set_pos(self.target_pos[0], self.target_pos[1])
"""
class Target(Geom):

    def __init__(self, env_data):
        self.herd_centre = env_data.herd_centre
        herd_target_radius = env_data.config.herd_target_radius
        crosshair_size = 10
        color = (0, 0, 0)
        self.vertical_bar = Part(rendering.Line((-crosshair_size - 1, 0), (crosshair_size, 0)))
        self.horizontal_bar = Part(rendering.Line((0, -crosshair_size - 1), (0, crosshair_size)))
        self.herd_circle = Part(rendering.make_circle(herd_target_radius, res=50, filled=False))

        self.vertical_bar.set_color(*color)
        self.horizontal_bar.set_color(*color)
        self.herd_circle.set_color(*color)

    def get_parts(self):
        return [self.vertical_bar.body, self.horizontal_bar.body, self.herd_circle.body]

    def update(self):
        self.horizontal_bar.set_pos(self.herd_centre[0], self.herd_centre[1])
        self.vertical_bar.set_pos(self.herd_centre[0], self.herd_centre[1])
        self.herd_circle.set_pos(self.herd_centre[0], self.herd_centre[1])
"""
