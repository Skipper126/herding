from abc import ABC


class RewardCounter(ABC):

    def reset(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError
