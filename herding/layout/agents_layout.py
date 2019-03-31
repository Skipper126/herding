from herding import opencl
from herding.data import EnvData


class AgentsLayout:

    def __init__(self, env_data: EnvData):
        self.env_data = env_data
        self.workers_count = env_data.config.dogs_count + env_data.config.sheep_count + 1
        self.layout_kernel = env_data.ocl.create_module('herding/layout/agents_layout.cl',
                                                        env_data.config.agents_layout,
                                                        [env_data.shared_buffers.dogs_positions,
                                                         env_data.shared_buffers.sheep_positions,
                                                         env_data.shared_buffers.target_position,
                                                         env_data.shared_buffers.seed])

    def set_up_agents(self):
        self.layout_kernel.run((self.workers_count,), (1,))
