
__global__ void cast_rays(Arrays *arrays)
{
    arrays->observation[threadIdx.x][threadIdx.y][0] = ((float)threadIdx.y) / blockDim.y;
    arrays->observation[threadIdx.x][threadIdx.y][1] = 0;
}
