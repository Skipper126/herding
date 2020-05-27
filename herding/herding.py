import gym
import numpy as np
from herding.data import create_env_data
from herding.agents import AgentsController
from herding.layout import AgentsLayout
from herding.reward import create_reward_counter
import warnings


class Herding(gym.Env):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, **params):
        self.env_data = create_env_data(params)
        self._check_params(self.env_data)
        self.reward_counter = create_reward_counter(self.env_data)
        self.agents_controller = AgentsController(self.env_data)
        self.agents_layout = AgentsLayout(self.env_data)
        self.viewer = None

    def step(self, action):
        self.agents_controller.move_agents(action)
        observation = self.agents_controller.get_observation()
        reward = self.reward_counter.get_reward()
        is_done = self.reward_counter.is_done()

        return observation, reward, is_done, {}

    def reset(self):
        self.agents_layout.set_up_agents()
        self.reward_counter.reset()
        observation = self.agents_controller.get_observation()

        return observation

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from .rendering.renderer import Renderer
            self.viewer = Renderer(self.env_data)

        self.viewer.set_text(self._get_debug_text())
        self.viewer.render()

    def seed(self, seed=None):
        warnings.warn('Calling seed() will not take an effect. Pass the seed as env parameter instead.')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    @property
    def action_space(self) -> gym.spaces.Box:
        '''Returns action space for a single dog.'''
        return gym.spaces.Box(low=-1, high=1, shape=(self.env_data.config.channels_count,))

    @property
    def observation_space(self) -> gym.spaces.Box:
        '''Returns observation space for a single dog.'''
        return gym.spaces.Box(low=0, high=1, shape=(self.env_data.config.rays_count,
                                                    self.env_data.config.channels_count))

    def _get_debug_text(self):
        reward_text = 'Reward: {:.2f}'.format(self.reward_counter.get_episode_reward())

        return reward_text

    @staticmethod
    def _check_params(env_data):
        max_workgroup_size = env_data.ocl.get_max_work_group_size()
        if max_workgroup_size < env_data.config.sheep_count:
            raise ValueError('Sheep count should be below ' + \
                             str(max_workgroup_size) + '.')
