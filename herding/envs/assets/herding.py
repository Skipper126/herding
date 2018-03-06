import gym
import numpy as np
import random
from gym import spaces
from .rendering.renderer import Renderer
from . import constants
from . import agents


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
            continuous_sheep_spread_rate=1,
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

        self.map_height = 1280
        self.map_width = 1024
        self.agent_radius = 5

        self.herd_centre_point = [0, 0]
        self.dog_list = None
        self.sheep_list = None
        self.dog_list, self.sheep_list = self._create_agents()
        self._set_agents_lists()
        self.reward_counter = RewardCounter(self)
        self.viewer = None
        self.agent_layout_function = AgentLayoutFunction.get_function(self.agent_layout)

    def step(self, action):
        for i, dog in enumerate(self.dog_list):
            dog.move(action[i])

        for sheep in self.sheep_list:
            sheep.move()

        state = self._get_state()
        reward = self.reward_counter.get_reward()
        is_done = self.reward_counter.is_done()

        return state, reward, is_done, {}

    def reset(self):
        self._set_up_agents()
        state = self._get_state()

        return state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Renderer(self)

        self.viewer.render()

    def seed(self, seed=None):
        pass

    @property
    def single_action_space(self):
        dim = 3 if self.rotation_mode is constants.RotationMode.FREE else 2
        single_action_space = spaces.Box(-1, 1, (dim,))
        return single_action_space

    @property
    def action_space(self):
        action_space = spaces.Tuple((self.single_action_space,) * self.dog_count)
        return action_space

    @property
    def single_observation_space(self):
        single_observation_space = spaces.Box(-1, 1, (2, self.ray_count))
        return single_observation_space

    @property
    def observation_space(self):
        observation_space = spaces.Tuple((self.single_observation_space,) * self.dog_count)
        return observation_space

    def _create_agents(self):
        dog_list = []
        sheep_list = []
        Sheep = agents.get_sheep_class(self.sheep_type)

        for i in range(self.dog_count):
            dog_list.append(agents.Dog(self))
        for i in range(self.sheep_count):
            sheep_list.append(Sheep(self))

        return dog_list, sheep_list

    def _set_agents_lists(self):
        for agent in self.dog_list + self.sheep_list:
            agent.set_lists(self.dog_list, self.sheep_list)

    def _get_state(self):
        state = []
        for dog in self.dog_list:
            state.append(dog.get_observation())
        return state

    def _update_herd_centre_point(self):
        self.herd_centre_point[0] = self.herd_centre_point[1] = 0
        for sheep in self.sheep_list:
            self.herd_centre_point[0] += sheep.x
            self.herd_centre_point[1] += sheep.y

        self.herd_centre_point[0] /= self.sheep_count
        self.herd_centre_point[1] /= self.sheep_count

    def _set_up_agents(self):
        self.agent_layout_function(self)


    # def _checkIfDone(self):
    #     if self.scatter < self.params.SCATTER_LEVEL:
    #         return True
    #
    #     return False
    #
    # def _scatter(self):
    #     self.herdCentrePoint[0] = self.herdCentrePoint[1] = 0
    #     for sheep in self.sheepList:
    #         self.herdCentrePoint[0] += sheep.x
    #         self.herdCentrePoint[1] += sheep.y
    #
    #     self.herdCentrePoint[0] /= self.sheepCount
    #     self.herdCentrePoint[1] /= self.sheepCount
    #
    #     self.previousScatter = self.scatter
    #     self.scatter = 0
    #     for sheep in self.sheepList:
    #         self.scatter += (sheep.x - self.herdCentrePoint[0]).__pow__(2) + (
    #             sheep.y - self.herdCentrePoint[1]).__pow__(2)
    #
    # def _reward(self):
    #     self._scatter()
    #     self.rewardValue = self.previousScatter - self.scatter
    #     if self.scatter < self.previousScatter:
    #         self.rewardValue.__neg__()
    #     if self.scatter < self.params.SCATTER_LEVEL:
    #         self.rewardValue = self.params.REWARD_FOR_HERDING
    #
    #     return self.rewardValue


class RewardCounter:

    def __init__(self, env: Herding):
        self.herd_centre_point = env.herd_centre_point
        self.sheep_list = env.sheep_list
        self.sheep_count = env.sheep_count
        self.sheep_type = env.sheep_type
        self.previous_scatter = 0
        self.scatter = 0
        self.reward_value = 0
        self.constants_scatter_counter = 0

    def is_done(self):
        return False

    def get_reward(self):
        return 0

    def _get_scatter(self):
        scatter = 0
        for sheep in self.sheep_list:
            scatter += pow(sheep.x - self.herd_centre_point[0],2) + \
                       pow(sheep.y - self.herd_centre_point[1],2)

        scatter /= self.sheep_count
        return 0


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
        padding = 5
        for agent in env.dog_list:
            x = random.randint(agent.radius + padding, env.map_width - agent.radius - padding)
            y = random.randint(agent.radius + padding, env.map_height - agent.radius - padding)
            agent.set_pos(x, y)

        w_padding = int(env.map_width / 6)
        h_padding = int(env.map_height / 6)
        for agent in env.sheep_list:
            x = random.randint(agent.radius + w_padding, env.map_width - agent.radius - w_padding)
            y = random.randint(agent.radius + h_padding, env.map_height - agent.radius - h_padding)
            agent.set_pos(x, y)

    @staticmethod
    def _layout2(env):
        # TODO
        pass
