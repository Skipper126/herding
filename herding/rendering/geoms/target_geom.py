from gym.envs.classic_control import rendering


class Target():
    BODY = 0

    def __init__(self, env_data):
        self.body = Part(rendering.make_circle(4, res=4))
        self.body.set_color(0, 0, 1)
        self.region = Part(rendering.make_circle(env_data.config.herd_target_radius, 30, False))
        self.region.set_color(0, 0, 0)

    def get_parts(self):
        return [self.body.body, self.region.body]

    def update(self, arrays):
        pos_x = arrays.target_position[0]
        pos_y = arrays.target_position[1]
        self.body.set_pos(pos_x, pos_y)
        self.region.set_pos(pos_x, pos_y)
