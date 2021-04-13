from herding import opencl, data, agents
import numpy as np

# def test_matrix_sorter_complete_sort():
#     env_data = _get_env_data()
#     matrix_buffer = env_data.shared_buffers.agents_matrix1
#     matrix_sorter = agents.MatrixSorter(env_data)
#
#     matrix = matrix_buffer.map_write()
#
#     np.copyto(matrix, random_matrix)
#     matrix_buffer.unmap()
#
#     matrix_sorter.sort(matrix_buffer)
#     gpu_sorted_matrix = matrix_buffer.map_read()
#     sorted_matrix = _sort(random_matrix)
#
#     assert np.array_equal(gpu_sorted_matrix, sorted_matrix)


def test_helper_sorter():
    matrix_size = 5
    input_matrix = _get_sample_matrix(size=matrix_size, direction='desc')
    expected_sorted_array = _get_sample_matrix(size=matrix_size, direction='asc')

    sorted_matrix = _sort(input_matrix)
    assert np.array_equal(sorted_matrix, expected_sorted_array)


# =================== Helpers =============================

def _sort_columns(matrix: np.ndarray, offset: int):
    for i in range(matrix.shape[0]):
        for j in range(int(matrix.shape[1] / 2)):
            j *= 2
            pos_x1 = matrix[i, j + offset, 0]
            pos_x2 = matrix[i, j + offset + 1, 0]
            if pos_x1 > pos_x2:
                tmp = np.copy(matrix[i, j + offset])
                matrix[i, j + offset] = matrix[i, j + offset + 1]
                matrix[i, j + offset + 1] = tmp

def _sort_rows(matrix: np.ndarray, offset: int):
    for j in range(matrix.shape[1]):
        for i in range(int(matrix.shape[0] / 2)):
            i *= 2
            pos_y1 = matrix[i + offset, j, 1]
            pos_y2 = matrix[i + offset + 1, j, 1]
            if pos_y1 > pos_y2:
                tmp = np.copy(matrix[i + offset, j])
                matrix[i + offset, j] = matrix[i + offset + 1, j]
                matrix[i + offset + 1, j] = tmp

def _sort(matrix: np.ndarray) -> np.ndarray:
    out_matrix = np.copy(matrix)
    for i in range(matrix.shape[0]):
        _sort_columns(out_matrix, 0)
        _sort_columns(out_matrix, 1)

    for i in range(matrix.shape[1]):
        _sort_rows(out_matrix, 0)
        _sort_rows(out_matrix, 1)

    return out_matrix

def _get_sample_matrix(size: int, direction: str) -> np.ndarray:
    list_range = list(range(size)) if direction == 'asc' else list(range(size - 1, -1, -1))

    value_matrix = np.array(np.meshgrid(list_range, list_range, indexing='ij')).T.reshape(-1, 2)
    zero_padding = np.zeros((size * size, 2))
    complete_matrix = np.hstack((value_matrix, zero_padding)).reshape((size, size, 4))

    return complete_matrix

def _get_env_data() -> data.EnvData:
    params = {
        'sheep_count': 300,
        'dogs_count': 5
    }
    env_data = data.EnvData(config=data.create_config(params))
    data.init_opencl(env_data)

    return env_data
