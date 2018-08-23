

__device__ void move_dogs(Arrays *arrays)
{
    arrays.dogs_positions[idx][0] += arrays.action[idx][0];
    arrays.dogs_positions[idx][1] += arrays.action[idx][1];
    arrays.dogs_rotations[idx] += arrays.action[idx][3];
}