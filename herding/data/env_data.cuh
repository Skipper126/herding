#define DOGS_COUNT $dogs_count
#define SHEEP_COUNT $sheep_count
#define HERD_TARGET_RADIUS $herd_target_radius
#define AGENT_RADIUS $agent_radius
#define MAX_MOVEMENT_SPEED $max_movement_speed
#define MAX_ROTATION_SPEED $max_rotation_speed
#define MAX_EPISODE_REWARD $max_episode_reward
#define RAYS_COUNT $rays_count
#define RAY_LENGTH $ray_length
#define FIELD_OF_VIEW $field_of_view
#define DOG_COLOR_R $dog_color_r
#define DOG_COLOR_G $dog_color_g
#define DOG_COLOR_B $dog_color_b
#define SHEEP_COLOR_R $sheep_color_r
#define SHEEP_COLOR_G $sheep_color_g
#define SHEEP_COLOR_B $sheep_color_b
#define TARGET_COLOR_R $target_color_r
#define TARGET_COLOR_G $target_color_g
#define TARGET_COLOR_B $target_color_b
#define DOG_BARK_COLOR_R $dog_bark_color_r
#define DOG_BARK_COLOR_G $dog_bark_color_g
#define DOG_BARK_COLOR_B $dog_bark_color_b

#define PI 3.14159

struct Arrays {
    float dogs_positions[DOGS_COUNT][2];
    float sheep_positions[SHEEP_COUNT][2];
    float observation[DOGS_COUNT][RAYS_COUNT][2][3];
    float rays_lengths[DOGS_COUNT][RAYS_COUNT];
    float dogs_rotations[DOGS_COUNT];
    float rand_values[SHEEP_COUNT];
    float target[2];
    float action[DOGS_COUNT][3];

};
