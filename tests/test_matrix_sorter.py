import pytest
from herding import data, agents
import tests.shared
import numpy as np
import math


# tests for single columns/rows pass are here until there is a way to do that in a single kernel invocation.

@pytest.mark.parametrize('offset', [0, 1])
def test_matrix_sorter_sort_columns_single_pass(matrix_sorter, matrix_buffer, unsorted_matrix, offset):
    matrix_sorter._sort_columns_single_pass(matrix_buffer, offset)
    sorted_matrix = matrix_buffer.map_read()

    expected_sorted_matrix = sort_columns_single_pass(unsorted_matrix, offset)
    assert np.array_equal(sorted_matrix, expected_sorted_matrix)


@pytest.mark.parametrize('offset', [0, 1])
def test_matrix_sorter_sort_rows_single_pass(matrix_sorter, matrix_buffer, unsorted_matrix, offset):
    matrix_sorter._sort_rows_single_pass(matrix_buffer, offset)
    sorted_matrix = matrix_buffer.map_read()

    expected_sorted_matrix = sort_rows_single_pass(unsorted_matrix, offset)
    assert np.array_equal(sorted_matrix, expected_sorted_matrix)


def test_matrix_sorter_sort_single_pass(matrix_sorter, matrix_buffer, expected_sorted_matrix_single_pass):
    matrix_sorter.sort_single_pass()
    sorted_matrix = matrix_buffer.map_read()

    assert np.array_equal(sorted_matrix, expected_sorted_matrix_single_pass)


def test_matrix_sorter_sort_complete(matrix_sorter, matrix_buffer, expected_sorted_matrix):
    matrix_sorter.sort_complete()
    sorted_matrix = matrix_buffer.map_read()

    assert np.array_equal(sorted_matrix, expected_sorted_matrix)


def test_helper_sorter(unsorted_matrix, expected_sorted_matrix):
    sorted_matrix = sort_complete(unsorted_matrix)

    assert np.array_equal(sorted_matrix, expected_sorted_matrix)


# =================== Fixtures =============================


@pytest.fixture
def unsorted_matrix(env_data):
    side_length = math.isqrt(env_data.config.dogs_count + env_data.config.sheep_count + 1)

    return _get_unsorted_matrix(side_length)


@pytest.fixture
def expected_sorted_matrix(unsorted_matrix):
    return sort_complete(unsorted_matrix)


@pytest.fixture
def expected_sorted_matrix_single_pass(unsorted_matrix):
    return sort_single_pass(unsorted_matrix)


@pytest.fixture
def env_data():
    params = {
        'sheep_count': 80,
        'dogs_count': 1
    }

    return tests.shared.get_env_data(params)


@pytest.fixture
def matrix_sorter(env_data):
    return agents.MatrixSorter(env_data)


@pytest.fixture
def matrix_buffer(env_data, matrix_sorter, unsorted_matrix):
    matrix_buffer = env_data.shared_buffers.input_matrix
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


def sort_single_pass(matrix: np.ndarray) -> np.ndarray:
    out_matrix = np.copy(matrix)
    out_matrix = sort_columns_single_pass(out_matrix, 0)
    out_matrix = sort_columns_single_pass(out_matrix, 1)
    out_matrix = sort_rows_single_pass(out_matrix, 0)
    out_matrix = sort_rows_single_pass(out_matrix, 1)

    return out_matrix


def sort_complete(matrix: np.ndarray) -> np.ndarray:
    out_matrix = np.copy(matrix)
    for i in range(matrix.shape[0]):
        out_matrix = sort_columns_single_pass(out_matrix, 0)
        out_matrix = sort_columns_single_pass(out_matrix, 1)

    for i in range(matrix.shape[1]):
        out_matrix = sort_rows_single_pass(out_matrix, 0)
        out_matrix = sort_rows_single_pass(out_matrix, 1)

    return out_matrix


def _get_unsorted_matrix(side_length: int) -> np.ndarray:
    list_range = list(range(side_length - 1, -1, -1))

    value_matrix = np.array(np.meshgrid(list_range, list_range, indexing='ij')).T.reshape(-1, 2)
    data_padding = np.arange(side_length * side_length * 6, dtype=np.float32).reshape((side_length * side_length, 6))
    complete_matrix = np.hstack((value_matrix, data_padding)).reshape((side_length, side_length, 8))

    return complete_matrix.astype(np.float32)


def _get_env_data(params) -> data.EnvData:
    env_data = data.EnvData(config=data.create_config(params))
    data.init_opencl(env_data)

    return env_data
