from herding.envs.assets.configuration.names import ConfigName as cn


class Agent:

    def __init__(self, env_data, index):
        self.radius = env_data.config[cn.AGENT_RADIUS]
        self.index = index
