
def dog_move(index, action):
    delta_x = action[0] * self.max_movement_speed
    delta_y = action[1] * self.max_movement_speed

    vec_length = math.sqrt(delta_x * delta_x + delta_y * delta_y)
    if vec_length > self.max_movement_speed:
        norm = self.max_movement_speed / vec_length
        delta_x *= norm
        delta_y *= norm

    if self.rotation_mode is RotationMode.FREE:
        self.rotation += action[2] * self.max_rotation_speed * DEG2RAD
        self.rotation = self.rotation % (2 * math.pi)
    else:
        self.rotation = np.arctan2(self.pos[coo.Y] - self.herd_centre[coo.Y],
                                   self.pos[coo.X] - self.herd_centre[coo.X]) + 90 * DEG2RAD

    cos_rotation = math.cos(self.rotation)
    sin_rotation = math.sin(self.rotation)
    self.pos[coo.X] += delta_x * cos_rotation + delta_y * sin_rotation
    self.pos[coo.Y] += delta_y * -cos_rotation + delta_x * sin_rotation