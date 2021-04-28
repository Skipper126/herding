from typing import Dict

import gym
from herding import data, agents, layout, reward
import warnings


class Herding(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, params: Dict=None):
        self.env_data = data.EnvData(config=data.create_config(params))
        # OpenCL and other modules are lazy loaded when calling reset() for the first time
        self.reward_counter: reward.RewardCounter
        self.agents_controller: agents.AgentsController
        self.agents_layout: layout.AgentsLayout
        self.viewer = None
        self.env_initialised = False

    def step(self, action):
        return (None, 0, False, None)
        self.agents_controller.move_agents(action)
        observation = self.agents_controller.get_observation()
        reward = self.reward_counter.get_reward()
        is_done = self.reward_counter.is_done()

        return observation, reward, is_done, {}

    def reset(self):
        if not self.env_initialised:
            self._init_env_modules()
            self.env_initialised = True

        self.agents_layout.set_up_agents()
        #self.reward_counter.reset()
        #observation = self.agents_controller.get_observation()

        return None

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
    def action_space(self) -> gym.spaces.MultiDiscrete:
        '''
        Returns action space for a single dog.
        Actions are:
        1) Movement: BACK[0], NOOP[1], FORWARD[2]
        2) Rotation: LEFT[0], NOOP[1], RIGHT[2]
        '''
        return gym.spaces.MultiDiscrete([3, 3])

    @property
    def observation_space(self) -> gym.spaces.Box:
        '''Returns observation space for a single dog.'''
        return gym.spaces.Box(low=0, high=1, shape=(self.env_data.config.rays_count,
                                                    self.env_data.config.channels_count))

    def _init_env_modules(self):
        data.init_opencl(self.env_data)
        self.reward_counter = reward.RewardCounter(self.env_data)
        self.agents_controller = agents.AgentsController(self.env_data)
        self.agents_layout = layout.AgentsLayout(self.env_data)

    def _get_debug_text(self):
        reward_text = 'Reward: {:.2f}'.format(self.reward_counter.get_episode_reward())

        return reward_text
