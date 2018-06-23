

class SimpleMove:

    def __init__(self, sheep):
        self.sheep = sheep

    def move(self):
        delta_x = 0
        delta_y = 0
        for dog in self.sheep.dog_list:
            distance = pow(pow((self.sheep.x - dog.x), 2) + pow((self.sheep.y - dog.y), 2), 0.5)
            if distance < 200:
                if distance < 50:
                    distance = 50
                delta_x += ((self.sheep.x - dog.x) / distance) * (200 - distance)
                delta_y += ((self.sheep.y - dog.y) / distance) * (200 - distance)

        if delta_x > 50 or delta_y > 50:
            if delta_x > delta_y:
                delta_y = delta_y / delta_x * 50
                delta_x = 50
            else:
                delta_x = delta_x / delta_y * 50
                delta_y = 50

        delta_x = delta_x / 50 * self.sheep.max_movement_speed
        delta_y = delta_y / 50 * self.sheep.max_movement_speed
        self.sheep.x += delta_x
        self.sheep.y += delta_y
