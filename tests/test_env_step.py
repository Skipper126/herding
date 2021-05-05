import itertools
import pytest
import tests.shared
import tests.test_matrix_sorter
from herding import data
#from herding.agents import AgentsController
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
    return env_data.shared_buffers.input_matrix


@pytest.fixture
def agents_controller(env_data):
    return None # AgentsController(env_data)


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

# agent structure
POS_X = 0
POS_Y = 1
ROT = 2
VEL = 3
TYPE = 4
AUX_ID = 5
AGENT_TYPE_SHEEP = 0
AGENT_TYPE_DOG = 1

PI = 3.141592
DEG2RAD = 0.01745329252


def step(input_matrix: np.ndarray,
         output_matrix: np.ndarray,
         actions: np.ndarray,
         observations: np.ndarray,
         reward: np.ndarray,
         env_data: data.EnvData):
    side_length = env_data.config.agents_matrix_side_length
    target_pos_x = env_data.config.target_x
    target_pos_y = env_data.config.target_y
    target_radius = 50
    scan_radius = env_data.config.scan_radius
    sight_distance = env_data.config.ray_length
    n_side_length = env_data.config.neighbours_matrix_side_length
    n_count = n_side_length**2
    delta_movement = np.zeros((side_length, side_length, 2,))


    for i, j, n_i, n_j in itertools.product(range(side_length), range(side_length), range(n_side_length),
                                            range(n_side_length)):
        n_i = n_i + i - scan_radius
        n_j = n_j + j - scan_radius

        if n_i < 0 or n_i >= side_length or n_j < 0 or n_j >= side_length:
            continue

        agent = input_matrix[i, j]
        n_agent = input_matrix[n_i, n_j]

        if i == n_i and j == n_j:  # processing self
            if agent[TYPE] == AGENT_TYPE_DOG:
                _process_action(env_data, i, j, output_matrix, actions, agent)
            else:
                pass#_process_reward(env_data, agent, target_pos_x, target_pos_y, target_radius, reward)
        elif _distance(agent, n_agent) < sight_distance:  # process neighbour
            if agent[TYPE] == AGENT_TYPE_DOG:
                pass#_process_observation(env_data, agent, n_agent, observations)
            else:
                _process_flock_behaviour(env_data, i, j, agent, n_agent, delta_movement)

    # sync point
    for i, j, n_i, n_j in itertools.product(range(side_length), range(side_length), range(n_side_length),
                                            range(n_side_length)):
        n_i = n_i + i - scan_radius
        n_j = n_j + j - scan_radius

        if n_i < 0 or n_i >= side_length or n_j < 0 or n_j >= side_length:
            continue

        agent = input_matrix[i, j]
        n_agent = input_matrix[n_i, n_j]

        if i == n_i and j == n_j:  # processing self
            if agent[TYPE] == AGENT_TYPE_SHEEP:
                output_matrix[i, j][POS_X] = agent[POS_X] + delta_movement[i, j, 0]
                output_matrix[i, j][POS_Y] = agent[POS_Y] + delta_movement[i, j, 1]
                output_matrix[i, j][ROT] = agent[ROT]
                output_matrix[i, j][VEL] = env_data.config.movement_speed
                output_matrix[i, j][TYPE] = 0
                output_matrix[i, j][AUX_ID] = 0


def _process_action(env_data, i, j, output_matrix, actions, agent):
    dog_id = int(agent[AUX_ID])
    action = actions[dog_id]
    delta_movement = (action[0] - 1) * env_data.config.movement_speed
    rotation = agent[ROT] + (action[1] - 1) * env_data.config.rotation_speed * DEG2RAD
    if rotation < 0:
        rotation = 2 * PI + rotation
    elif rotation > 2 * PI:
        rotation = rotation - 2 * PI

    cos_rotation = np.math.cos(-rotation)
    sin_rotation = np.math.sin(-rotation)

    output_matrix[i, j][POS_X] = agent[POS_X] + delta_movement * sin_rotation
    output_matrix[i, j][POS_Y] = agent[POS_Y] + delta_movement * cos_rotation
    output_matrix[i, j][ROT] = rotation
    output_matrix[i, j][VEL] = env_data.config.movement_speed
    output_matrix[i, j][TYPE] = 1
    output_matrix[i, j][AUX_ID] = dog_id


def _process_reward(env_data, agent, target_pos_x, target_pos_y, target_radius, reward):
    if _distance_xy(agent, target_pos_x, target_pos_y) < target_radius:
        reward[0] += 1


def _process_observation(env_data, agent, n_agent, observations: np.ndarray):
    # need to provide zeroed observation array
    dog_id = agent[AUX_ID]
    observation = observations[dog_id]
    n_angle = np.math.atan2(agent[POS_Y] - n_agent[POS_Y], agent[POS_X] - n_agent[POS_X] + PI)
    n_count = observation.shape[0]
    ray_id = int(n_count * (n_angle - agent[ROT]) / PI)
    if ray_id < 0 or ray_id > n_count:
        return
    observation[ray_id][:] = [0, 1, 0]


def _process_flock_behaviour(env_data, i, j, agent, n_agent, delta_movement):
    delta_x = 0
    delta_y = 0

    pos_x_diff = agent[POS_X] - n_agent[POS_X]
    pos_y_diff = agent[POS_Y] - n_agent[POS_Y]

    delta_x = pos_x_diff
    delta_y = pos_y_diff

    delta_x = (delta_x / 50) * env_data.config.movement_speed
    delta_y = (delta_y / 50) * env_data.config.movement_speed

    delta_movement[i, j, 0] += delta_x
    delta_movement[i, j, 1] += delta_y


def _distance(agent1, agent2) -> float:
    return _distance_xy(agent1, agent2[POS_X], agent2[POS_Y])


def _distance_xy(agent1, pos_x, pos_y) -> float:
    diff_x = agent1[POS_X] - pos_x
    diff_y = agent1[POS_Y] - pos_y

    return np.math.sqrt(diff_x ** 2 + diff_y ** 2)
