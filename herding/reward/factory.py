from herding.data import EnvData
from herding.reward.furthest_sheep import FurthestSheepRewardCounter
from herding.reward.medium_distance import MediumDistanceRewardCounter
from herding.reward import RewardCounter


reward_counter_type = {
    'furthest_sheep': FurthestSheepRewardCounter,
    'medium_distance': MediumDistanceRewardCounter
}


def create_reward_counter(env_data: EnvData) -> RewardCounter:
    return reward_counter_type[env_data.config.reward_type](env_data)
