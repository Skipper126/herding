#define PI 3.141592

__kernel void set_up_agents(__global float (*dogs_positions)[3],
                            __global float (*sheep_positions)[2],
                            __global float (*target_position),
                            __global unsigned int (*seed))
{
    int id = get_global_id(0);

    if (id < DOGS_COUNT)
    {
        dogs_positions[id][0] = 500;
        dogs_positions[id][1] = 100;
        dogs_positions[id][2] = 0;
    }
    else if (id - DOGS_COUNT < SHEEP_COUNT)
    {
        sheep_positions[id - DOGS_COUNT][0] = 500;
        sheep_positions[id - DOGS_COUNT][1] = 500;
    }
    else
    {
        target_position[0] = 500;
        target_position[1] = 900;
    }
}
