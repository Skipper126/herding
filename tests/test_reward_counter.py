import pytest
from herding.reward import RewardCounter













# =================== Helpers =============================

def _get_env_data_mock():
    env_data = {
        'config': {
            'sheep_count': 3,
            'max_episode_reward': 100,
            'herd_target_radius': 100,

        }
    }
