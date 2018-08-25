from .geom import *
from gym.envs.classic_control import rendering


class SheepGeom(Geom):
    BODY = 0

    def __init__(self, env_data, sheep_index):
        self.sheep_radius = env_data.config.agent_radius
        self.sheep_pos = env_data.sheep_positions[sheep_index]
        self.body = Part(rendering.make_circle(self.sheep_radius, res=10))
        self.body.set_color(181 / 255, 185 / 255, 215 / 255)

    def get_parts(self):
        return [self.body.body]

    def update(self):
        self.body.set_pos(self.sheep_pos[0], self.sheep_pos[1])
