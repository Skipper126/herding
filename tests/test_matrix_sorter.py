import pytest
from herding import data, agents
import numpy as np


@pytest.mark.parametrize('offset', [0, 1])
def test_matrix_sorter_sort_columns_single_pass(matrix_sorter, matrix_buffer, unsorted_matrix, offset):
    matrix_sorter.sort_columns_single_pass(matrix_buffer, offset)
    sorted_matrix = matrix_buffer.map_read()

    expected_sorted_matrix = sort_columns_single_pass(unsorted_matrix, offset)
    assert np.array_equal(sorted_matrix, expected_sorted_matrix)


@pytest.mark.parametrize('offset', [0, 1])
def test_matrix_sorter_sort_rows_single_pass(matrix_sorter, matrix_buffer, unsorted_matrix, offset):
    matrix_sorter.sort_rows_single_pass(matrix_buffer, offset)
    sorted_matrix = matrix_buffer.map_read()

    expected_sorted_matrix = sort_rows_single_pass(unsorted_matrix, offset)
    assert np.array_equal(sorted_matrix, expected_sorted_matrix)


def test_helper_sorter(unsorted_matrix, sorted_matrix):
    sort_result_matrix = sort(unsorted_matrix)

    assert np.array_equal(sort_result_matrix, sorted_matrix)


# =================== Fixtures =============================


@pytest.fixture
def unsorted_matrix():
    return _get_sample_matrix(size=101, direction='desc')


@pytest.fixture
def sorted_matrix():
    return _get_sample_matrix(size=101, direction='asc')


@pytest.fixture
def env_data():
    return _get_env_data()


@pytest.fixture
def matrix_sorter(env_data):
    return agents.MatrixSorter(env_data)


@pytest.fixture
def matrix_buffer(env_data, matrix_sorter, unsorted_matrix):
    matrix_buffer = env_data.ocl.create_buffer(unsorted_matrix.shape, unsorted_matrix.dtype)
    matrix = matrix_buffer.map_write()
    np.copyto(matrix, unsorted_matrix)
    matrix_buffer.unmap()

    return matrix_buffer


# =================== Helpers =============================


def sort_columns_single_pass(input_matrix: np.ndarray, offset: int) -> np.ndarray:
    matrix = np.copy(input_matrix)
    for i in range(matrix.shape[0]):
        for j in range(int(matrix.shape[1] / 2)):
            j *= 2
            pos_x1 = matrix[i, j + offset, 0]
            pos_x2 = matrix[i, j + offset + 1, 0]
            if pos_x1 > pos_x2:
                tmp = np.copy(matrix[i, j + offset])
                matrix[i, j + offset] = matrix[i, j + offset + 1]
                matrix[i, j + offset + 1] = tmp

    return matrix


def sort_rows_single_pass(input_matrix: np.ndarray, offset: int) -> np.ndarray:
    matrix = np.copy(input_matrix)
    for j in range(matrix.shape[1]):
        for i in range(int(matrix.shape[0] / 2)):
            i *= 2
            pos_y1 = matrix[i + offset, j, 1]
            pos_y2 = matrix[i + offset + 1, j, 1]
            if pos_y1 > pos_y2:
                tmp = np.copy(matrix[i + offset, j])
                matrix[i + offset, j] = matrix[i + offset + 1, j]
                matrix[i + offset + 1, j] = tmp

    return matrix

def sort(matrix: np.ndarray) -> np.ndarray:
    out_matrix = np.copy(matrix)
    for i in range(matrix.shape[0]):
        out_matrix = sort_columns_single_pass(out_matrix, 0)
        out_matrix = sort_columns_single_pass(out_matrix, 1)

    for i in range(matrix.shape[1]):
        out_matrix = sort_rows_single_pass(out_matrix, 0)
        out_matrix = sort_rows_single_pass(out_matrix, 1)

    return out_matrix


def _get_sample_matrix(size: int, direction: str) -> np.ndarray:
    list_range = list(range(size)) if direction == 'asc' else list(range(size - 1, -1, -1))

    value_matrix = np.array(np.meshgrid(list_range, list_range, indexing='ij')).T.reshape(-1, 2)
    zero_padding = np.zeros((size * size, 2))
    complete_matrix = np.hstack((value_matrix, zero_padding)).reshape((size, size, 4))

    return complete_matrix.astype(np.float32)


def _get_env_data() -> data.EnvData:
    params = {
        'sheep_count': 300,
        'dogs_count': 5
    }
    env_data = data.EnvData(config=data.create_config(params))
    data.init_opencl(env_data)

    return env_data
