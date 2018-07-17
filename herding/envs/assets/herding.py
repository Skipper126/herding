import gym
import numpy as np
from herding.envs.assets import configuration
from herding.envs.assets import data
from herding.envs.assets.controller import AgentsController
from herding.envs.assets.reward import RewardCounter
from herding.envs.assets.layout import AgentsLayout


class Herding(gym.Env):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, **kwargs):
        self.env_data = self._get_env_data(kwargs)

        self.reward_counter = RewardCounter(self.env_data)
        self.agents_controller = AgentsController(self.env_data)
        self.agents_layout = AgentsLayout(self.env_data)
        self.viewer = None

    def step(self, action):
        self.agents_controller.move_dogs(action)
        self.agents_controller.move_sheep()

        self.reward_counter.update_herd_centre()
        state = self.agents_controller.get_observation()
        reward = self.reward_counter.get_reward()
        is_done = self.reward_counter.is_done()

        return state, reward, is_done, {
            "scatter": self.reward_counter.scatter
        }

    def reset(self):
        self.agents_layout.set_up_agents()
        self.reward_counter.reset()
        state = self.agents_controller.get_observation()

        return state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from .rendering.renderer import Renderer
            self.viewer = Renderer(self)

        self.viewer.render()

    def seed(self, seed=None):
        seed = seed if seed is not None else 100
        np.random.seed(seed)

    def close(self):
        self.viewer.close()
        self.agents_controller.close()

    @staticmethod
    def _get_env_data(params):
        config = configuration.get_default_configuration()
        config.update(params)
        shared_data = data.SharedData(params)

        env_data = data.EnvData(config, shared_data)

        return env_data
