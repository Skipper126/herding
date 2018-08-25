

__device__ void move_dogs(Arrays *arrays)
{
    arrays->dogs_positions[threadIdx.x][0] += arrays->action[threadIdx.x][0] * 10;
    arrays->dogs_positions[threadIdx.x][1] -= arrays->action[threadIdx.x][1] * 10;
    arrays->dogs_rotations[threadIdx.x] += arrays->action[threadIdx.x][2];
}
