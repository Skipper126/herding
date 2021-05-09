#define PI 3.141592
#define DEG2RAD 0.01745329252
#define POS_X s0
#define POS_Y s1
#define ROT s2
#define VEL s3
#define TYPE s4
#define AUX_ID s5
#define AGENT_TYPE_SHEEP 0
#define AGENT_TYPE_DOG 1

#define N_SIDE_LENGTH (SCAN_RADIUS * 2 + 1)


float get_distance(float x1, float y1, float x2, float y2)
{
    float x_diff = x1 - x2;
    float y_diff = y1 - y2;
    return sqrt((x_diff * x_diff) + (y_diff * y_diff));
}

void process_flock_behaviour(int i, int j, float8 agent, float8 n_agent, __local float *delta_movement)
{
    float pos_x_diff = agent.POS_X - n_agent.POS_X;
    float pos_y_diff = agent.POS_Y - n_agent.POS_Y;
    float distance = get_distance(agent.POS_X, agent.POS_Y, n_agent.POS_X, n_agent.POS_Y);
    float max_distance = n_agent.TYPE == AGENT_TYPE_DOG ? 80 : 10;
    float min_distance = n_agent.TYPE == AGENT_TYPE_DOG ? 40 : 5;

    if (distance < max_distance)
    {
        if (distance < min_distance)
        {
            distance = min_distance;
        }

        delta_movement[0] += (pos_x_diff / distance) * (max_distance - distance);
        delta_movement[1] += (pos_y_diff / distance) * (max_distance - distance);
    }
}

void process_action(int i, int j,
                    __global float8 (*output_matrix)[AGENTS_MATRIX_SIDE_LENGTH],
                    __global int (*actions)[2],
                    float8 agent)
{
    int dog_id = (int)agent.AUX_ID;
    __global int *action = actions[dog_id];
    float delta_movement = (action[0] - 1) * MOVEMENT_SPEED;
    float rotation = agent.ROT + (action[1] - 1) * ROTATION_SPEED * DEG2RAD;

    if (rotation < 0)
    {
        rotation = 2 * PI + rotation;
    }
    else if (rotation > 2 * PI)
    {
        rotation = rotation - 2 * PI;
    }

    float cos_rotation = cos(-rotation);
    float sin_rotation = sin(-rotation);

    output_matrix[i][j].POS_X = agent.POS_X + delta_movement * sin_rotation;
    output_matrix[i][j].POS_Y = agent.POS_Y + delta_movement * cos_rotation;
    output_matrix[i][j].ROT = rotation;
    output_matrix[i][j].VEL = MOVEMENT_SPEED;
    output_matrix[i][j].TYPE = 1;
    output_matrix[i][j].AUX_ID = dog_id;
}

void process_observation(int i, int j, int n_i, int n_j, float8 agent, float8 n_agent, __global float (*observations)[RAYS_COUNT][3])
{
    float x = agent.POS_X;
    float y = agent.POS_Y;
    float n_x = (i == n_i && j == n_j) ? TARGET_X : n_agent.POS_X;
    float n_y = (i == n_i && j == n_j) ? TARGET_Y : n_agent.POS_Y;
    float n_angle = atan2(y - n_y, x - n_x) + PI;
    float first_ray_angle = agent.ROT;
    float last_ray_angle = agent.ROT < PI ? agent.ROT + PI : agent.ROT - PI;
    int n_k = (int)(((n_angle - first_ray_angle) / PI) * RAYS_COUNT);

    if (!(n_k >= 0 && n_k < RAYS_COUNT))
    {
        n_k = RAYS_COUNT - (int)(((last_ray_angle - n_angle) / PI) * RAYS_COUNT);
    }

    if (n_k >= 0 && n_k < RAYS_COUNT)
    {
        int dog_id = (int)agent.AUX_ID;

        if (i == n_i && j == n_j)
        {
            observations[dog_id][n_k][0] = 0;
            observations[dog_id][n_k][1] = 0;
            observations[dog_id][n_k][2] = 1;                
        }
        else
        {
            
            observations[dog_id][n_k][0] = 0;
            observations[dog_id][n_k][1] = 1;
            observations[dog_id][n_k][2] = 0;                
        }
    }    
}

__kernel void env_step(__global float8 (*input_matrix)[AGENTS_MATRIX_SIDE_LENGTH],
                       __global float8 (*output_matrix)[AGENTS_MATRIX_SIDE_LENGTH],
                       __global int (*actions)[2],
                       __global float (*observations)[RAYS_COUNT][3])
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_local_id(2);
    int n_i = (int)(k / N_SIDE_LENGTH) + i - SCAN_RADIUS;
    int n_j = k - (int)(k / N_SIDE_LENGTH) * N_SIDE_LENGTH + j - SCAN_RADIUS;

    __local float delta_movement[2];
    __local float dog_neighbour;

    float8 agent = input_matrix[i][j];

    if (i == n_i && j == n_j)
    {
        delta_movement[0] = 0;
        delta_movement[1] = 0;
        dog_neighbour = 0;
    }
    
    if (agent.TYPE == AGENT_TYPE_DOG)
    {
        int dog_id = (int)agent.AUX_ID;
        observations[dog_id][k][0] = 0;
        observations[dog_id][k][1] = 0;
        observations[dog_id][k][2] = 0;

        n_i = n_i + (int)(round(cos(agent.ROT)) * SCAN_RADIUS);
        n_j = n_j - (int)(round(sin(agent.ROT)) * SCAN_RADIUS);
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    if (n_i < 0 || n_i >= AGENTS_MATRIX_SIDE_LENGTH ||
        n_j < 0 || n_j >= AGENTS_MATRIX_SIDE_LENGTH)
    {
        goto sync;
    }

    float8 n_agent = input_matrix[n_i][n_j];

    if (i == n_i && j == n_j)
    {
        if (agent.TYPE == AGENT_TYPE_DOG)
        {
            process_action(i, j, output_matrix, actions, agent);
        }
    }
    else if (agent.TYPE == AGENT_TYPE_SHEEP)
    {
        process_flock_behaviour(i, j, agent, n_agent, delta_movement);
    }


    
    if (agent.TYPE == AGENT_TYPE_DOG && get_distance(agent.POS_X, agent.POS_Y, n_agent.POS_X, n_agent.POS_Y) < RAY_LENGTH)
    {
        process_observation(i, j, n_i, n_j, agent, n_agent, observations);
    }

    sync:
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // if (agent.TYPE == AGENT_TYPE_DOG)
    // {
    //     int dog_id = (int)agent.AUX_ID;
    //     if (observations[dog_id][k][0] == 1)
    //     {
    //         observations[dog_id][k][0] = 0;
    //         observations[dog_id][k][1] = 0;
    //         observations[dog_id][k][2] = 1;            
    //     }
    // }

    if (n_i < 0 || n_i >= AGENTS_MATRIX_SIDE_LENGTH ||
        n_j < 0 || n_j >= AGENTS_MATRIX_SIDE_LENGTH)
    {
        return;
    }

    if (i == n_i && j == n_j)
    {
        if (agent.TYPE == AGENT_TYPE_SHEEP)
        {
            float max_distance = n_agent.TYPE == AGENT_TYPE_DOG ? 80 : 10;
            float min_distance = n_agent.TYPE == AGENT_TYPE_DOG ? 40 : 5;
            if (delta_movement[0] > min_distance || delta_movement[1] > min_distance)
            {
                if (delta_movement[0] > delta_movement[1])
                {
                    delta_movement[1] = (delta_movement[1] / delta_movement[0]) * min_distance;
                    delta_movement[0] = min_distance;
                }
                else
                {
                    delta_movement[0] = (delta_movement[0] / delta_movement[1]) * min_distance;
                    delta_movement[1] = min_distance;
                }
            }

            delta_movement[0] = (delta_movement[0] / min_distance) * MOVEMENT_SPEED;
            delta_movement[1] = (delta_movement[1] / min_distance) * MOVEMENT_SPEED;


            output_matrix[i][j].POS_X = agent.POS_X + delta_movement[0];
            output_matrix[i][j].POS_Y = agent.POS_Y + delta_movement[1];
            output_matrix[i][j].ROT = agent.ROT;
            output_matrix[i][j].VEL = MOVEMENT_SPEED;
            output_matrix[i][j].TYPE = 0;
            output_matrix[i][j].AUX_ID = dog_neighbour;
        }
    }

    if (agent.TYPE == AGENT_TYPE_DOG && !(i == n_i && j == n_j))
    {
        output_matrix[n_i][n_j].AUX_ID = 10;
    }

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

    delta_x = (delta_x / dog_min_distance) * MOVEMENT_SPEED;
    delta_y = (delta_y / dog_min_distance) * MOVEMENT_SPEED;

    sheep_positions[id][0] += delta_x;
    sheep_positions[id][1] += delta_y;
}
/*
__kernel void move_sheep_complex(__global float (*dogs_positions)[3],
                                 __global float (*sheep_positions)[2],
                                 __global float (*seed))
{
    int id = get_global_id(0);
    unsigned int seed_value = seed[id];

    move_sheep_simple(dogs_positions, sheep_positions);
    seed[id] = seed_value;

    if (rand(&seed_value, 100) > 10)
        return;

    sheep_positions[id][0] += 1;
    sheep_positions[id][1] += 1;

}
*/
