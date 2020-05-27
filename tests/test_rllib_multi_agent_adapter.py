from typing import Dict, cast
import numpy as np
import pytest
from utils.rllib_multiagent_adapter import MultiAgentHerding
from herding import Herding
pytest.importorskip('ray.rllib')

base_test_case = {
    'env_config': {
        'dogs_count': 3
    },
    'action': np.stack((
        np.full((3,), 0),
        np.full((3,), 1),
        np.full((3,), 2)
    )),
    'observation': np.stack((
        np.full((128, 3), 0),
        np.full((128, 3), 1),
        np.full((128, 3), 2)
    )),
    'reward': 10,
    'is_done': False,
    'rllib_action': {
        "dog_0": np.full((3,), 0),
        "dog_1": np.full((3,), 1),
        "dog_2": np.full((3,), 2)
    },
    'rllib_observation': {
        "dog_0": np.full((128, 3), 0),
        "dog_1": np.full((128, 3), 1),
        "dog_2": np.full((128, 3), 2)
    },
    'rllib_reward': {
        "dog_0": 10,
        "dog_1": 10,
        "dog_2": 10
    },
    'rllib_dones': {
        "dog_0": False,
        "dog_1": False,
        "dog_2": False,
        '__all__': False
    }
}

test_cases = [
    base_test_case,
    {   # Swapping agents keys in rllib_action dict should not influence transformation
        **base_test_case,
        'rllib_action': {
            "dog_1": np.full((3,), 1),
            "dog_0": np.full((3,), 0),
            "dog_2": np.full((3,), 2)
        }
    },
]

@pytest.mark.parametrize('test_config', test_cases)
def test_rllib_adapter_step(test_config):
    env = MultiAgentHerding(env=_get_herding_mock(test_config))

    rllib_observation, rllib_reward, rllib_dones, _ = env.step(test_config['rllib_action'])

    assert _compare_rllib_dicts(rllib_observation, test_config['rllib_observation'])
    assert _compare_rllib_dicts(rllib_reward, test_config['rllib_reward'])
    assert _compare_rllib_dicts(rllib_dones, test_config['rllib_dones'])


@pytest.mark.parametrize('test_config', test_cases)
def test_rllib_adapter_reset(test_config):
    env = MultiAgentHerding(env=_get_herding_mock(test_config))

    rllib_observation = env.reset()
    assert _compare_rllib_dicts(rllib_observation, test_config['rllib_observation'])



# ===================================== Helpers =====================================

def _compare_rllib_dicts(dict_a: Dict[str, np.array], dict_b: Dict[str, np.array]) -> bool:
    k1 = sorted(dict_a.keys())
    k2 = sorted(dict_b.keys())

    if k1 != k2:
        return False

    return all([np.array_equal(dict_a[k], dict_b[k]) for k in k1])

def _get_herding_mock(test_config: Dict) -> Herding:

    class HerdingMock():
        env_data = _create_env_data_property(test_config)
        def step(self, action):
            assert np.array_equal(action, test_config['action'])
            state = test_config['observation']
            reward = test_config['reward']
            is_done = test_config['is_done']

            return state, reward, is_done, {}

        def reset(self):
            return test_config['observation']

    return cast(Herding, HerdingMock())


def _create_env_data_property(test_config: Dict) -> object:
    return type('',(),{'config': type('',(),{'dogs_count': test_config['env_config']['dogs_count']})})
