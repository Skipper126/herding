from herding import data, opencl
import numpy as np


class MatrixSorter:

    def __init__(self, env_data: data.EnvData):
        self.env_data = env_data

    def sort_columns_single_pass(self, matrix: opencl.Buffer, offset: int):
        offset_buffer = self.env_data.ocl.create_buffer((1,), np.int32)
        offset_mapped = offset_buffer.map_write()
        np.copyto(offset_mapped, offset)
        offset_buffer.unmap()
        module = self.env_data.ocl.create_module('herding/agents/matrix_sorter.cl',
                                                  'sort_columns_single_pass',
                                                  [matrix, offset_buffer])
        module.run((matrix.shape[0], int(matrix.shape[1] / 2)))

    def sort_rows_single_pass(self, matrix: opencl.Buffer, offset: int):
        offset_buffer = self.env_data.ocl.create_buffer((1,), np.int32)
        offset_mapped = offset_buffer.map_write()
        np.copyto(offset_mapped, offset)
        offset_buffer.unmap()
        module = self.env_data.ocl.create_module('herding/agents/matrix_sorter.cl',
                                                  'sort_rows_single_pass',
                                                  [matrix, offset_buffer])
        module.run((int(matrix.shape[0] / 2), matrix.shape[1]))

    def sort_complete(self, matrix: opencl.Buffer):
        for i in range(matrix.shape[0]):
            self.sort_columns_single_pass(matrix, 0)
            self.sort_columns_single_pass(matrix, 1)

        for i in range(matrix.shape[1]):
            self.sort_rows_single_pass(matrix, 0)
            self.sort_rows_single_pass(matrix, 1)
