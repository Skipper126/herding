from abc import ABC


class RewardCounter(ABC):

    def reset(self):
        raise NotImplementedError

    def get_reward(self) -> float:
        raise NotImplementedError

    def get_episode_reward(self) -> float:
        raise NotImplementedError

    def is_done(self) -> bool:
        raise NotImplementedError
