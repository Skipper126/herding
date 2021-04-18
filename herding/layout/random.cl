#include "herding/opencl/rand.h"
#define PI 3.141592

__kernel void set_up_agents(__global float8 (*agents_matrix)[(DOGS_COUNT + SHEEP_COUNT + 1) / 2],
                            __global unsigned int (*seed)[(DOGS_COUNT + SHEEP_COUNT + 1) / 2])
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    unsigned int seed_value = seed[i][j];
    float pos_x = rand(&seed_value, AGENTS_LAYOUT_WIDTH);
    float pos_y = rand(&seed_value, AGENTS_LAYOUT_HEIGHT);
    float direction = rand(&seed_value, (int)(2 * PI));

    // set random pos X(0), Y(1) and direction(2). Leave velocity at 0(3).
    // set type(4) to 0 for sheep, 1 for dog, 2 for target
    // set unused values(5, 6, 7) to 0

    agents_matrix[i][j].x = pos_x;
    agents_matrix[i][j].y = pos_y;
    agents_matrix[i][j].z = direction;
    agents_matrix[i][j].w = 0;
    agents_matrix[i][j].s5 = 0.0;
    agents_matrix[i][j].s6 = 0.0;
    agents_matrix[i][j].s7 = 0.0;

    if (i + j < DOGS_COUNT)
    {
        agents_matrix[i][j].s4 = 1;
    }
    else if (i + j - DOGS_COUNT < SHEEP_COUNT)
    {
        agents_matrix[i][j].s4 = 0;
    }
    else
    {
        agents_matrix[i][j].s4 = 2;
    }

    seed[i][j] = seed_value;
}
