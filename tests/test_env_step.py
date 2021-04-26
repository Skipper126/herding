import itertools

import pytest
import tests.shared
import tests.test_matrix_sorter
from herding import data
from herding.agents import AgentsController
from herding.layout import AgentsLayout
import numpy as np


def test_dogs_move(agents_controller, agents_matrix_buffer, input_action, expected_matrix_after_move):
    agents_controller.move_agents(input_action)
    agents_matrix = agents_matrix_buffer.map_read()

    np.testing.assert_array_equal(agents_matrix, expected_matrix_after_move)


@pytest.fixture
def input_action():
    # 1) Movement: BACK[0], NOOP[1], FORWARD[2]
    # 2) Rotation: LEFT[0], NOOP[1], RIGHT[2]
    return np.array([[2, 2], [2, 1]])


@pytest.fixture
def agents_matrix_buffer(env_data):
    agents_layout = AgentsLayout(env_data)
    agents_layout.set_up_agents()
    return env_data.shared_buffers.current_agents_matrix


@pytest.fixture
def agents_controller(env_data):
    return AgentsController(env_data)


@pytest.fixture
def env_data():
    params = {
        'sheep_count': 7,
        'dogs_count': 2,
        'agents_layout': 'random',
        'seed': 100,
    }

    return tests.shared.get_env_data(params)


# =================== Iteration implementation =============================

POS_X = 0
POS_Y = 1
DIR = 2
VEL = 3
TYPE = 4
AUX_ID = 5
AGENT_TYPE_SHEEP = 0
AGENT_TYPE_DOG = 1


def step(input_matrix: np.ndarray,
         output_matrix: np.ndarray,
         actions: np.ndarray,
         observation: np.ndarray,
         reward: np.ndarray,
         env_data: data.EnvData):
    side_length = env_data.config.agents_matrix_side_length
    target_pos_x = 100
    target_pos_y = 100
    target_radius = 50
    scan_radius = 2
    sight_distance = 600
    n_side_length = scan_radius * 2 + 1

    for i, j, n_i, n_j in itertools.product(range(side_length), range(side_length), range(n_side_length),
                                            range(n_side_length)):
        n_i = n_i + i - scan_radius
        n_j = n_j + i - scan_radius

        if n_i < 0 or n_i > side_length or n_j < 0 or n_j > side_length:
            continue

        agent = input_matrix[i, j]
        n_agent = input_matrix[n_i, n_j]

        if i == n_i and j == n_j:  # processing self
            if agent[TYPE] == AGENT_TYPE_DOG:
                _process_action(i, j, output_matrix, actions, agent)
            else:
                _process_reward(agent, target_pos_x, target_pos_y, target_radius, reward)
        elif _distance(agent, n_agent) < sight_distance:  # process neighbour
            if agent[TYPE] == AGENT_TYPE_DOG:
                _process_observation(agent, n_agent, observation)
            else:
                _process_flock_behaviour(agent, n_agent, output_matrix)


def _process_action(i, j, output_matrix, actions, agent):
    pass


def _process_reward(agent, target_pos_x, target_pos_y, target_radius, reward):
    pass


def _process_observation(agent, n_agent, observation):
    pass


def _process_flock_behaviour(agent, n_agent, output_matrix):
    pass


def _distance(agent1, agent2) -> int:
    pass
