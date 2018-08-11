from .geom import *
from openaigym.envs.classic_control import rendering
from openaigym.envs.assets.configuration.names import ConfigName as cn
from openaigym.envs.assets.configuration.constants import Coordinate as coo


class SheepGeom(Geom):
    BODY = 0

    def __init__(self, env_data, sheep_index):
        self.sheep_radius = env_data.config[cn.AGENT_RADIUS]
        self.sheep_pos = env_data.shared_data.sheep_positions[sheep_index]
        self.body = Part(rendering.make_circle(self.sheep_radius, res=50))
        self.body.set_color(181 / 255, 185 / 255, 215 / 255)

    def get_parts(self):
        return [self.body.body]

    def update(self):
        self.body.set_pos(self.sheep_pos[coo.X], self.sheep_pos[coo.Y])
