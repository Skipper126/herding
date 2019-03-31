
float rand(int *seed, int range) // 1 <= *seed < m
{
    long const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1

    *seed = (*seed * a) % m;

    return (*seed) % range;
}
