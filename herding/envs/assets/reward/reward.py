import math


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

    def reset(self):
        pass

    def update_herd_centre(self):
        pass
