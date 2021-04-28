import pytest
import tests.shared
import tests.test_matrix_sorter
from herding.layout import AgentsLayout
import numpy as np


def test_random_layout(agents_layout, agents_matrix_buffer, expected_random_layout):
    agents_layout.set_up_agents()
    agents_matrix = agents_matrix_buffer.map_read()

    np.testing.assert_array_equal(agents_matrix, expected_random_layout)


@pytest.fixture
def agents_matrix_buffer(env_data):
    return env_data.shared_buffers.current_agents_matrix


@pytest.fixture
def agents_layout(env_data):
    return AgentsLayout(env_data)


@pytest.fixture
def env_data():
    params = {
        'sheep_count': 46,
        'dogs_count': 35,
        'agents_layout': 'random',
        'seed': 100,
        'agents_layout_width': 1000,
        'agents_layout_height': 1000,
    }

    return tests.shared.get_env_data(params)


@pytest.fixture
def expected_random_layout(expected_random_layout_unsorted):
    return tests.test_matrix_sorter.sort_complete(expected_random_layout_unsorted)


@pytest.fixture
def expected_random_layout_unsorted(env_data) -> np.ndarray:
    agents_matrix = np.empty(env_data.shared_buffers.current_agents_matrix.shape, np.float32)
    # seed array is used by every worker during its computations. Following similar approach
    seed_array = np.copy(env_data.shared_buffers.seed.map_read())
    env_data.shared_buffers.seed.unmap()
    dogs_count = env_data.config.dogs_count
    sheep_count = env_data.config.sheep_count
    range_x = env_data.config.agents_layout_height
    range_y = env_data.config.agents_layout_width
    # Same precision as in kernel definition
    pi = 3.141592

    # set random pos X(0), Y(1) and direction(2). Leave velocity at 0(3).
    # set type(4) to 0 for sheep, 1 for dog, 2 for target
    # set unused values(5, 6, 7) to 0
    for i in range(agents_matrix.shape[0]):
        for j in range(agents_matrix.shape[1]):
            agents_matrix[i, j, 0] = _rand(seed_array, i, j, range_x)
            agents_matrix[i, j, 1] = _rand(seed_array, i, j, range_y)
            agents_matrix[i, j, 2] = _rand(seed_array, i, j, int(2 * pi))
            agents_matrix[i, j, 3] = 0
            agents_matrix[i, j, 4] = 1 if i * agents_matrix.shape[1] + j < dogs_count else 0
            agents_matrix[i, j, 5] = i * agents_matrix.shape[1] + j if agents_matrix[i, j, 4] == 1 else 0
            agents_matrix[i, j, 6:8] = 0

    return agents_matrix.astype(np.float32)



# This must match herding/opencl/rand.h
def _rand(seed_array: np.ndarray, i: int, j: int, range: int):
    seed_value = np.uint64(seed_array[i, j]) * np.uint64(214013) + np.uint64(2531011)
    new_seed = seed_value >> np.uint64(16)
    seed_array[i, j] = new_seed

    return new_seed % np.uint64(range)
