
__device__ void move_sheep_simple(Arrays *arrays)
{
    float delta_x = 0;
    float delta_y = 0;
    float *sheep_pos = arrays->sheep_positions[threadIdx.x];
    float dog_max_distance = 200.0;
    float dog_min_distance = 50.0;
    for (int i = 0; i < DOGS_COUNT; ++i)
    {
        float *dog_pos = arrays->dogs_positions[i];
        float pos_x_diff = sheep_pos[0] - dog_pos[0];
        float pos_y_diff = sheep_pos[1] - dog_pos[1];
        float distance = sqrtf(powf(pos_x_diff, 2) +
                               powf(pos_y_diff, 2));

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

    sheep_pos[0] += delta_x;
    sheep_pos[1] += delta_y;
}
