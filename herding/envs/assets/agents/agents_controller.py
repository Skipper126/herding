from abc import ABC
import abc

class AgentsController(ABC):

    @abc.abstractmethod
    def move_agents(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
