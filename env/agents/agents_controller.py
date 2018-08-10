

class AgentsController:

    def __init__(self, env_data):
        self.kernels = Cuda.get_kernels('dir')

    def move_agents(self, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
