from herding import data, opencl
import numpy as np


class MatrixSorter:

    def __init__(self, env_data: data.EnvData):
        self.env_data = env_data
        self.matrix = env_data.shared_buffers.current_agents_matrix

    def sort_single_pass(self):
        # TODO This should work as a single kernel invocation
        self._sort_columns_single_pass(self.matrix, 0)
        self._sort_columns_single_pass(self.matrix, 1)
        self._sort_rows_single_pass(self.matrix, 0)
        self._sort_rows_single_pass(self.matrix, 1)

    def sort_complete(self):
        for i in range(self.matrix.shape[0]):
            self._sort_columns_single_pass(self.matrix, 0)
            self._sort_columns_single_pass(self.matrix, 1)

        for i in range(self.matrix.shape[1]):
            self._sort_rows_single_pass(self.matrix, 0)
            self._sort_rows_single_pass(self.matrix, 1)

    def _sort_columns_single_pass(self, matrix: opencl.Buffer, offset: int):
        offset_buffer = self.env_data.ocl.create_buffer((1,), np.int32)
        offset_mapped = offset_buffer.map_write()
        np.copyto(offset_mapped, offset)
        offset_buffer.unmap()
        module = self.env_data.ocl.create_module('herding/agents/matrix_sorter.cl',
                                                  'sort_columns_single_pass',
                                                  [matrix, offset_buffer])
        module.run((matrix.shape[0], int(matrix.shape[1] / 2)))

    def _sort_rows_single_pass(self, matrix: opencl.Buffer, offset: int):
        offset_buffer = self.env_data.ocl.create_buffer((1,), np.int32)
        offset_mapped = offset_buffer.map_write()
        np.copyto(offset_mapped, offset)
        offset_buffer.unmap()
        module = self.env_data.ocl.create_module('herding/agents/matrix_sorter.cl',
                                                  'sort_rows_single_pass',
                                                  [matrix, offset_buffer])
        module.run((int(matrix.shape[0] / 2), matrix.shape[1]))
