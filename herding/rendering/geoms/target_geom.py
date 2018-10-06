from .geom import *
from gym.envs.classic_control import rendering


class Target(Geom):
    BODY = 0

    def __init__(self, env_data):
        self.target_pos = env_data.host_arrays.target
        self.body = Part(rendering.make_circle(4, res=4))
        self.body.set_color(0, 0, 1)

    def get_parts(self):
        return [self.body.body]

    def update(self):
        self.body.set_pos(self.target_pos[0], self.target_pos[1])
