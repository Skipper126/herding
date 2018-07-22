from ..agents_controller import AgentsController
import pycuda.driver as cuda

class GpuAgentsController(AgentsController):

    def __init__(self, env_data):
        module = kernels.get_module()

    def move_agents(self, action):
        pass

    def get_observation(self):
        pass

    def close(self):
        pass

    def _get_kernels(self):
        pass


