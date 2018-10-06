from .geom import *
from gym.envs.classic_control import rendering


class SheepGeom(Geom):
    BODY = 0

    def __init__(self, env_data, sheep_index):
        self.sheep_radius = env_data.config.agent_radius
        self.sheep_pos = env_data.host_arrays.sheep_positions[sheep_index]
        self.body = Part(rendering.make_circle(self.sheep_radius, res=8))
        self.body.set_color(0.7, 0.7, 0.7)

    def get_parts(self):
        return [self.body.body]

    def update(self):
        self.body.set_pos(self.sheep_pos[0], self.sheep_pos[1])
