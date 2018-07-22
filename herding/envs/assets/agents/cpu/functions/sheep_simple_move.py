def move(self):
    delta_x = 0
    delta_y = 0
    for dog_pos in self.dogs_positions:
        distance = pow(pow((self.pos[coo.X] - dog_pos[coo.X]), 2) +
                       pow((self.pos[coo.Y] - dog_pos[coo.Y]), 2), 0.5)
        if distance < 200:
            if distance < 50:
                distance = 50
            delta_x += ((self.pos[coo.X] - dog_pos[coo.X]) / distance) * (200 - distance)
            delta_y += ((self.pos[coo.Y] - dog_pos[coo.Y]) / distance) * (200 - distance)

    if delta_x > 50 or delta_y > 50:
        if delta_x > delta_y:
            delta_y = delta_y / delta_x * 50
            delta_x = 50
        else:
            delta_x = delta_x / delta_y * 50
            delta_y = 50

    delta_x = delta_x / 50 * self.max_movement_speed
    delta_y = delta_y / 50 * self.max_movement_speed
    self.pos[coo.X] += delta_x
    self.pos[coo.Y] += delta_y