from herding import opencl
from herding.data import EnvData


class AgentsLayout:

    def __init__(self, env_data: EnvData):
        self.env_data = env_data
        self.workers_count = env_data.config.dogs_count + env_data.config.sheep_count + 1
        self.layout_kernel = opencl.create_module(env_data,
                                                  'herding/layout/layout.cl',
                                                  env_data.config.agents_layout)

    def set_up_agents(self):
        self.layout_kernel.run(self.workers_count)
