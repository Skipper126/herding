from .geom import *
from gym.envs.classic_control import rendering
from herding.data import get_color_tuple_from_config


class SheepGeom(Geom):
    BODY = 0

    def __init__(self, env_data, sheep_index):
        self.sheep_radius = env_data.config.agent_radius
        self.sheep_pos = env_data.sheep_positions[sheep_index]
        self.body = Part(rendering.make_circle(self.sheep_radius, res=8))
        #sheep_color = get_color_tuple_from_config(env_data.config.sheep_color)
        self.body.set_color(*(env_data.config.sheep_color_r,
                              env_data.config.sheep_color_g,
                              env_data.config.sheep_color_b,))

    def get_parts(self):
        return [self.body.body]

    def update(self):
        self.body.set_pos(self.sheep_pos[0], self.sheep_pos[1])
