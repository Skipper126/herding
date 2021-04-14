
// agent position struct

__kernel void sort_columns(__global float4 (*agents_matrix)[2], int offset)
{
    int row_id = get_global_id(0);
    int column_id = get_global_id(1) * 2 + offset;

    float4 agent1 = agents_matrix[column_id][row_id];
    float4 agent2 = agents_matrix[column_id + 1][row_id];

    if (if agent1.x > agent2.x)
    {
        agents_matrix[column_id][row_id] = agent2;
        agents_matrix[column_id][row_id] = agent1;
    }
}

__kernel void sort_rows(__global float4 (*agents_matrix)[2], int offset)
{
    int row_id = get_global_id(0) * 2 + offset;
    int column_id = get_global_id(1);

    float4 agent1 = agents_matrix[column_id][row_id];
    float4 agent2 = agents_matrix[column_id + 1][row_id];

    if (if agent1.x > agent2.x)
    {
        agents_matrix[column_id][row_id] = agent2;
        agents_matrix[column_id][row_id] = agent1;
    }
}
