
// Defines are taken directly from config
#define PI 3.14159

struct Arrays {
    float rays_lengths[DOGS_COUNT][RAYS_COUNT];
    float dogs_positions[DOGS_COUNT][2];
    float dogs_rotations[DOGS_COUNT];
    float sheep_positions[SHEEP_COUNT][2];
    float target_position[2];
    float observation[DOGS_COUNT][RAYS_COUNT][CHANNELS_COUNT];
    float action[DOGS_COUNT][3];
    int   seed[DOGS_COUNT + SHEEP_COUNT + 1];
    float common_output;
};
