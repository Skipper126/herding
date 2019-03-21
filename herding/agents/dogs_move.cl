

__device__ void move_dogs(Arrays *arrays)
{
    arrays->dogs_positions[threadIdx.x][0] += arrays->action[threadIdx.x][0] * 10;
    arrays->dogs_positions[threadIdx.x][1] -= arrays->action[threadIdx.x][1] * 10;
    arrays->dogs_rotations[threadIdx.x] += arrays->action[threadIdx.x][2];
    float rotation =  arrays->dogs_rotations[threadIdx.x];
    if (rotation < 0)
    {
        arrays->dogs_rotations[threadIdx.x] = 2 * PI + rotation;
    }
    if (rotation > 2 * PI)
    {
        arrays->dogs_rotations[threadIdx.x] = rotation - 2 * PI;
    }
}
