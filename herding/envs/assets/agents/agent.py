from herding.envs.assets.configuration.constants import ConfigName as cn


class Agent:

    def __init__(self, env_data, index):
        self.radius = env_data.config[cn.AGENT_RADIUS]
        self.index = index
        self.dogs_positions = env_data.shared_data.dogs_positions
        self.sheep_positions = env_data.shared_data.sheep_positions

