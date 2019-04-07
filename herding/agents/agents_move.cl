#define PI 3.141592
#define DEG2RAD 0.01745329252


__kernel void move_dogs(__global float (*dogs_positions)[3],
                        __global float (*action)[3])
{
    int id = get_global_id(0);
    float delta_x = action[id][0] * MAX_MOVEMENT_SPEED;
    float delta_y = action[id][1] * MAX_MOVEMENT_SPEED;

    float move_vector = sqrt(delta_x * delta_x + delta_y * delta_y);
    if (move_vector > MAX_MOVEMENT_SPEED)
    {
        float scale_down = MAX_MOVEMENT_SPEED / move_vector;
        delta_x *= scale_down;
        delta_y *= scale_down;
    }

    float rotation = dogs_positions[id][2] + action[id][2] * MAX_ROTATION_SPEED * DEG2RAD;
    if (rotation < 0)
    {
        rotation = 2 * PI + rotation;
    }
    else if (rotation > 2 * PI)
    {
        rotation = rotation - 2 * PI;
    }

    float cos_rotation = cos(rotation);
    float sin_rotation = sin(rotation);

    dogs_positions[id][0] += delta_x * cos_rotation + delta_y * sin_rotation;
    dogs_positions[id][1] += delta_y * (-cos_rotation) + delta_x * sin_rotation;
    dogs_positions[id][2] = rotation;
}

__kernel void move_sheep_simple(__global float (*dogs_positions)[3],
                                __global float (*sheep_positions)[2])
{
    int id = get_global_id(0);
    float delta_x = 0;
    float delta_y = 0;

    float sheep_pos_x = sheep_positions[id][0];
    float sheep_pos_y = sheep_positions[id][1];
    float dog_max_distance = SHEEP_FLEE_DISTANCE;
    float dog_min_distance = 50.0;

    for (int i = 0; i < DOGS_COUNT; i++)
    {
        float dog_pos_x = dogs_positions[i][0];
        float dog_pos_y = dogs_positions[i][1];
        float pos_x_diff = sheep_pos_x - dog_pos_x;
        float pos_y_diff = sheep_pos_y - dog_pos_y;
        float distance = sqrt(pow(pos_x_diff, 2) +
                               pow(pos_y_diff, 2));

        if (distance < dog_max_distance)
        {
            if (distance < dog_min_distance)
                distance = dog_min_distance;

            delta_x += (pos_x_diff / distance) *
                       (dog_max_distance - distance);
            delta_y += (pos_y_diff / distance) *
                       (dog_max_distance - distance);
        }
    }

    if (delta_x > dog_min_distance || delta_y > dog_min_distance)
    {
        if (delta_x > delta_y)
        {
            delta_y = (delta_y / delta_x) * dog_min_distance;
            delta_x = dog_min_distance;
        }
        else
        {
            delta_x = (delta_x / delta_y) * dog_min_distance;
            delta_y = dog_min_distance;
        }
    }

    delta_x = (delta_x / dog_min_distance) * MAX_MOVEMENT_SPEED;
    delta_y = (delta_y / dog_min_distance) * MAX_MOVEMENT_SPEED;

    sheep_positions[id][0] += delta_x;
    sheep_positions[id][1] += delta_y;
}
