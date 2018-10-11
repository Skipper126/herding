from herding import data
from herding.reward import RewardCounter
from herding.reward.furthest_sheep import FurthestSheepRewardCounter
from typing import Dict


def create_reward_counter(env_data: data.EnvData) -> RewardCounter:
    counters: Dict[str, RewardCounter] = {
        'furthest_sheep': FurthestSheepRewardCounter
    }
    RewardCounterType = counters[env_data.config.reward_type]
    return RewardCounterType(env_data)
