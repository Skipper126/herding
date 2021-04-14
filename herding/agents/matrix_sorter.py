from herding import data, opencl


class MatrixSorter:

    def __init__(self, env_data: data.EnvData):
        pass

    def sort_columns_single_pass(self, matrix: opencl.Buffer, offset: int):
        pass

    def sort_rows_single_pass(self, matrix: opencl.Buffer, offset: int):
        pass

    def sort_complete(self, matrix: opencl.Buffer):
        pass
