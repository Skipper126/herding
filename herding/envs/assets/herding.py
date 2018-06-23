import gym
import random
from gym import spaces
from . import constants
from . import agents
import math
import numpy as np

class Herding(gym.Env):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(
            self,
            dog_count=1,
            sheep_count=3,
            agent_layout=constants.AgentLayout.RANDOM,
            sheep_type=constants.SheepType.SIMPLE,
            max_movement_speed=5,
            max_rotation_speed=90,
            ray_count=180,
            ray_length=600,
            field_of_view=180,
            rotation_mode=constants.RotationMode.FREE,
    ):
        self.dog_count = dog_count
        self.sheep_count = sheep_count
        self.agent_layout = agent_layout
        self.sheep_type = sheep_type
        self.max_movement_speed = max_movement_speed
        self.max_rotation_speed = max_rotation_speed
        self.ray_count = ray_count
        self.ray_length = ray_length
        self.field_of_view = field_of_view
        self.rotation_mode = rotation_mode

        self.map_width = 800
        self.map_height = 600
        self.agent_radius = 10

        self.herd_target_radius = 100
        self.max_episode_reward = 100

        self.herd_centre_point = np.zeros(2)

        self.dogs_positions = np.zeros((self.dog_count, 2))
        self.sheep_positions = np.zeros((self.sheep_count, 2))

        self.reward_counter = RewardCounter(self)
        self.agents_shared_state = AgentsSharedState(self)
        self.agent_controller = AgentController(self)

        self.agents_shared_state.link_arrays(self.agents_shared_state.dogs_positions, self.dogs_positions)
        self.agents_shared_state.link_arrays(self.agents_shared_state.sheep_positions, self.sheep_positions)



        self.viewer = None
        self.agent_layout_function = AgentLayoutFunction.get_function(self.agent_layout)

    def step(self, action):
        self.agent_controller.move_dogs(action)
        self.agent_controller.move_sheep()

        self._update_herd_centre_point()
        state = self.agent_controller.get_observation()
        reward = self.reward_counter.get_reward()
        is_done = self.reward_counter.is_done()

        return state, reward, is_done, {
            "scatter": self.reward_counter.scatter
        }

    def reset(self):
        self._set_up_agents()
        self.reward_counter.reset()
        state = self.agent_controller.get_observation()

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
        random.seed(seed)

    def close(self):
        self.viewer.close()

    def _update_herd_centre_point(self):
        self.herd_centre_point = np.mean(self.sheep_positions, axis=0)

    def _set_up_agents(self):
        self.agent_layout_function(self)


class RewardCounter:

    def __init__(self, env):
        self.herd_centre_point = env.herd_centre_point

        self.sheep_list = env.sheep_list
        self.sheep_count = env.sheep_count
        self.sheep_type = env.sheep_type

        self.previous_scatter = 0
        self.scatter = 0
        self.first_scatter = 0
        self.first_iteration = True

        self.herd_target_radius = env.herd_target_radius
        self.max_episode_reward = env.max_episode_reward
        self.agent_radius = env.agent_radius

    def is_done(self):
        for sheep in self.sheep_list:
            distance = self._get_distance(sheep)

            if distance > self.herd_target_radius - self.agent_radius:
                return False

        self.first_iteration = True
        return True

    def get_reward(self):
        self.previous_scatter = self.scatter
        self.scatter = self._get_scatter()

        if self.first_iteration:
            self.first_iteration = False
            self.previous_scatter = self.scatter
            self.first_scatter = self.scatter

        return ((self.previous_scatter - self.scatter) * self.max_episode_reward) / self.first_scatter

    def _get_scatter(self):
        scatter = 0
        for sheep in self.sheep_list:
            scatter += max(self._get_distance(sheep) - self.herd_target_radius, 0)

        scatter /= self.sheep_count
        return scatter

    def _get_distance(self, sheep):
        return math.sqrt(pow(sheep.x - self.herd_centre_point[0], 2) + \
                       pow(sheep.y - self.herd_centre_point[1], 2))


class AgentLayoutFunction:

    @staticmethod
    def get_function(agent_layout):
        return{
            constants.AgentLayout.RANDOM : AgentLayoutFunction._random,
            constants.AgentLayout.LAYOUT1 : AgentLayoutFunction._layout1,
            constants.AgentLayout.LAYOUT2 : AgentLayoutFunction._layout2
        }[agent_layout]

    @staticmethod
    def _random(env):
        padding = 5
        for agent in env.dog_list + env.sheep_list:
            x = random.randint(agent.radius + padding, env.map_width - agent.radius - padding)
            y = random.randint(agent.radius + padding, env.map_height - agent.radius - padding)
            agent.set_pos(x, y)

    @staticmethod
    def _layout1(env):
        sheep_padding = 5
        for agent in env.sheep_list:
            x = random.randint(agent.radius + sheep_padding, env.map_width - agent.radius - sheep_padding)
            y = random.randint(agent.radius + sheep_padding + 200, env.map_height - agent.radius - sheep_padding)
            agent.set_pos(x, y)

        for i, agent in enumerate(env.dog_list):
            x = (i + 1) * (env.map_width / (env.dog_count + 1))
            y = 0
            agent.set_pos(x, y)

    @staticmethod
    def _layout2(env):
        # TODO
        pass
