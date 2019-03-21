from herding import data, opencl


class AgentsLayout:

    def __init__(self, env_data: data.EnvData):
        self.env_data = env_data
        self.workers_count = env_data.config.dogs_count + env_data.config.sheep_count + 1
        self.layout_kernel = opencl.create_module(env_data.ocl,
                                                  'herding/layout/layout.cl',
                                                  env_data.config.agents_layout)

    def set_up_agents(self):
        self.layout_kernel.run(self.workers_count)
