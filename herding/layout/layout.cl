#include "herding/data/env_data.h"
#include "herding/opencl/random.h"

__kernel random_layout(Arrays *arrays)
{
    int id = get_global_id(0);
    int *seed = arrays->seed + id;
    float x_pos = rand(seed, AGENTS_LAYOUT_RANGE);
    float y_pos = rand(seed, AGENTS_LAYOUT_RANGE);

    if (id < DOGS_COUNT)
    {
        float rotation = rand(seed, 2 * PI);
        arrays->dogs_positions[id][0] = x_pos;
        arrays->dogs_positions[id][1] = y_pos;
        arrays->dogs_rotations[id] = rotation;
    }
    else if (id < SHEEP_COUNT)
    {
        id -= DOGS_COUNT;
        arrays->sheep_positions[id][0] = x_pos;
        arrays->sheep_positions[id][1] = x_pos;
    }
    else
    {
        arrays->target_position[0] = x_pos;
        arrays->target_position[1] = y_pos;
    }
}