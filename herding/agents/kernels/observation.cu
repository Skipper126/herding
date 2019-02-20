#define DOG 0
#define SHEEP 1
#define TARGET 2
#define DEG2RAD 0.01745329252


__device__ void clear_observation(Arrays *arrays)
{
    for (int i = 0; i < 3; ++i)
    {
        arrays->observation[threadIdx.x][threadIdx.y][i] = 0;
    }

    arrays->rays_lengths[threadIdx.x][threadIdx.y] = 1;
}

__device__ float get_distance(float x1, float y1, float x2, float y2)
{
    float x_diff = x1 - x2;
    float y_diff = y1 - y2;
    return sqrtf((x_diff * x_diff) + (y_diff * y_diff));
}

__global__ void get_observation(Arrays *arrays)
{
    int dog_index = threadIdx.x;
    int ray_index = threadIdx.y;
    clear_observation(arrays);
    float dog_pos_x = arrays->dogs_positions[dog_index][0];
    float dog_pos_y = arrays->dogs_positions[dog_index][1];
    float ray_angle = arrays->dogs_rotations[dog_index] + (((float)threadIdx.y / RAYS_COUNT) * PI);
    float min_distance = RAY_LENGTH;
    if (ray_angle > 2 * PI)
    {
        ray_angle = ray_angle - 2 * PI;
    }
    for (int i = 0; i < SHEEP_COUNT; ++i)
    {
        float agent_pos_x = arrays->sheep_positions[i][0];
        float agent_pos_y = arrays->sheep_positions[i][1];
        float distance = get_distance(dog_pos_x, dog_pos_y, agent_pos_x, agent_pos_y);
        
        if (distance < min_distance)
        {
            float angle = (atan2f(dog_pos_y - agent_pos_y, dog_pos_x - agent_pos_x) + PI);

            if (fabsf(angle - ray_angle) < atanf(AGENT_RADIUS / distance))
            {
                min_distance = distance;
                arrays->observation[dog_index][ray_index][SHEEP] = 1;
                arrays->rays_lengths[dog_index][ray_index] = distance / RAY_LENGTH;
            }
        }
    }

    for (int i = 0; i < DOGS_COUNT; ++i)
    {
        if (i == dog_index)
            continue;

        float agent_pos_x = arrays->dogs_positions[i][0];
        float agent_pos_y = arrays->dogs_positions[i][1];
        float distance = get_distance(dog_pos_x, dog_pos_y, agent_pos_x, agent_pos_y);

        if (distance < min_distance)
        {
            float angle = (atan2f(dog_pos_y - agent_pos_y, dog_pos_x - agent_pos_x) + PI);

            if (fabsf(angle - ray_angle) < atanf(AGENT_RADIUS / distance))
            {
                min_distance = distance;
                arrays->observation[dog_index][ray_index][DOG] = 1;
                arrays->rays_lengths[dog_index][ray_index] = distance / RAY_LENGTH;
            }
        }
    }

    float target_pos_x = arrays->target[0];
    float target_pos_y = arrays->target[1];
    float distance = get_distance(dog_pos_x, dog_pos_y, target_pos_x, target_pos_y);
    if (distance < RAY_LENGTH)
    {
        float angle = (atan2f(dog_pos_y - target_pos_y, dog_pos_x - target_pos_x) + PI);

        if (fabsf(angle - ray_angle) < atanf(AGENT_RADIUS / distance))
        {
            arrays->observation[dog_index][ray_index][TARGET] = 1;
        }
    }

}
