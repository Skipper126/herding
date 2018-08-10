from gym.envs.classic_control import rendering


class Part:

    def __init__(self, body):
        self.body = body
        self.transform = rendering.Transform()
        self.body.add_attr(self.transform)

    def set_pos(self, x, y):
        self.transform.set_translation(x, y)

    def set_rotation(self, rotation):
        self.transform.set_rotation(rotation)

    def set_scale(self, x, y):
        self.transform.set_scale(x, y)

    def set_color(self, r, g, b):
        self.body.set_color(r, g, b)


class Geom:

    def update(self):
        raise NotImplementedError

    def get_parts(self):
        raise NotImplementedError
