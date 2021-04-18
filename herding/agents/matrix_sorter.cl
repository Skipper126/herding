

__kernel void sort_columns_single_pass(__global float8 (*agents_matrix)[(DOGS_COUNT + SHEEP_COUNT + 1) / 2], __global int *offset)
{
    int i = get_global_id(0);
    int j = get_global_id(1) * 2 + *offset;

    float8 agent1 = agents_matrix[i][j];
    float8 agent2 = agents_matrix[i][j + 1];

    if (agent1.x > agent2.x)
    {
        agents_matrix[i][j] = agent2;
        agents_matrix[i][j + 1] = agent1;
    }
}

__kernel void sort_rows_single_pass(__global float8 (*agents_matrix)[(DOGS_COUNT + SHEEP_COUNT + 1) / 2], __global int *offset)
{
    int i = get_global_id(0) * 2 + *offset;
    int j = get_global_id(1);

    float8 agent1 = agents_matrix[i][j];
    float8 agent2 = agents_matrix[i + 1][j];

    if (agent1.y > agent2.y)
    {
        agents_matrix[i][j] = agent2;
        agents_matrix[i + 1][j] = agent1;
    }
}
