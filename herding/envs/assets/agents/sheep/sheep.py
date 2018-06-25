from herding.envs.assets.agents.agent import Agent


class Sheep(Agent):

    def __init__(self, env):
        super().__init__(env)

        self.max_movement_speed = env.max_movement_speed

    def move(self):
        delta_x = 0
        delta_y = 0
        for dog in self.dog_list:
            distance = pow(pow((self.x - dog.x), 2) + pow((self.y - dog.y), 2), 0.5)
            if distance < 200:
                if distance < 50:
                    distance = 50
                delta_x += ((self.x - dog.x) / distance) * (200 - distance)
                delta_y += ((self.y - dog.y) / distance) * (200 - distance)

        if delta_x > 50 or delta_y > 50:
            if delta_x > delta_y:
                delta_y = delta_y / delta_x * 50
                delta_x = 50
            else:
                delta_x = delta_x / delta_y * 50
                delta_y = 50

        delta_x = delta_x / 50 * self.max_movement_speed
        delta_y = delta_y / 50 * self.max_movement_speed
        self.x += delta_x
        self.y += delta_y
