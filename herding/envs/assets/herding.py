import gym
import numpy as np
from herding import constants
from herding.envs.assets.params import Params
from herding.envs.assets.multiprocessing import SharedData
from herding.envs.assets.controller import Controller
from herding.envs.assets.reward import RewardCounter


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
            herd_target_radius=100,
            max_episode_reward=100,
            ray_count=180,
            ray_length=600,
            field_of_view=180,
            rotation_mode=constants.RotationMode.FREE,
    ):
        self.params = Params()
        self.params.dog_count = dog_count
        self.params.sheep_count = sheep_count
        self.params.agent_layout = agent_layout
        self.params.sheep_type = sheep_type
        self.params.max_movement_speed = max_movement_speed
        self.params.max_rotation_speed = max_rotation_speed
        self.params.herd_target_radius = herd_target_radius
        self.params.max_episode_reward = max_episode_reward
        self.params.ray_count = ray_count
        self.params.ray_length = ray_length
        self.params.field_of_view = field_of_view
        self.params.rotation_mode = rotation_mode

        self.map_width = 800
        self.map_height = 600
        self.agent_radius = 10

        self.shared_data = SharedData(self.params)

        self.herd_centre = self.shared_data.herd_centre
        self.dogs_positions = self.shared_data.dogs_positions
        self.sheep_positions = self.shared_data.sheep_positions

        self.reward_counter = RewardCounter(self.params)
        self.agent_controller = Controller(self.params)

        self.viewer = None
        self.agent_layout_function = AgentLayoutFunction.get_function(self.params.agent_layout)

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
