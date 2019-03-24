from herding.data import EnvData
from herding.reward.furthest_sheep import FurthestSheepRewardCounter
from herding.reward import RewardCounter


reward_counter_type = {
    'furthest_sheep': FurthestSheepRewardCounter
}


def create_reward_counter(env_data: EnvData) -> RewardCounter:
    return reward_counter_type[env_data.config.reward_type](env_data)
