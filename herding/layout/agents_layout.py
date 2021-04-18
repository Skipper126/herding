from herding.data import EnvData


class AgentsLayout:

    def __init__(self, env_data: EnvData):
        self.env_data = env_data
        matrix_side_lenght = int((env_data.config.dogs_count + env_data.config.sheep_count + 1) / 2)
        self.workers_shape = (matrix_side_lenght, matrix_side_lenght)
        self.layout_kernel = env_data.ocl.create_module('herding/layout/' + env_data.config.agents_layout + '.cl',
                                                        'set_up_agents',
                                                        [env_data.shared_buffers.current_agents_matrix,
                                                         env_data.shared_buffers.seed])

    def set_up_agents(self):
        self.layout_kernel.run(self.workers_shape)
