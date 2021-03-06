from herding import agents
from herding.data import EnvData


class AgentsLayout:

    def __init__(self, env_data: EnvData):
        self.env_data = env_data
        self.matrix_sorter = agents.MatrixSorter(env_data)
        agents_matrix_side_length = env_data.config.agents_matrix_side_length
        self.workers_shape = (agents_matrix_side_length, agents_matrix_side_length)
        self.layout_kernel = env_data.ocl.create_module('herding/layout/' + env_data.config.agents_layout + '.cl',
                                                        'set_up_agents',
                                                        [env_data.shared_buffers.input_matrix,
                                                         env_data.shared_buffers.seed])

    def set_up_agents(self):
        self.layout_kernel.run(self.workers_shape)
        self.matrix_sorter.sort_complete()
