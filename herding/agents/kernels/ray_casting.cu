
__global__ void cast_rays(Arrays *arrays)
{
    arrays->rays[threadIdx.x][threadIdx.y][0] = ((float)threadIdx.y) / blockDim.y;
    arrays->rays[threadIdx.x][threadIdx.y][1] = 0;
}
