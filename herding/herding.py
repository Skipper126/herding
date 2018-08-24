import gym
import numpy as np
from herding.data import create_env_data
from herding.agents import AgentsController
from herding.layout import AgentsLayout
from herding.reward import RewardCounter


class Herding(gym.Env):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, **params):
        self.env_data = create_env_data(params)
        self.reward_counter = RewardCounter(self.env_data)
        self.agents_controller = AgentsController(self.env_data)
        self.agents_layout = AgentsLayout(self.env_data)
        self.viewer = None

    def step(self, action):
        self.agents_controller.move_agents(action)
        self.reward_counter.update_herd_centre()
        state = self.agents_controller.get_observation()
        reward = self.reward_counter.get_reward()
        is_done = self.reward_counter.is_done()

        return state, reward, is_done, {}

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
            self.viewer = Renderer(self.env_data)

        self.viewer.render()

    def seed(self, seed=None):
        seed = seed if seed is not None else 100
        np.random.seed(seed)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
