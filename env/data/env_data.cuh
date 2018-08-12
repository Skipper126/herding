#pragma once

#define DOGS_COUNT $dogs_count
#define SHEEP_COUNT $sheep_count
#define RAYS_COUNT $rays_count

struct Config {
    int dogs_count;
    int sheep_count;
    int herd_target_radius;
    int rotation_mode;
    int agent_radius;
    int max_movement_speed;
    int max_rotation_speed;
    int max_episode_reward;
    int rays_count;
    int ray_length;
    int field_of_view;
};

struct EnvData {
    Config config;
    float dogs_positions[DOGS_COUNT][2];
    float sheep_positions[SHEEP_COUNT][2];
    float herd_centre[2];
    float observation[DOGS_COUNT][RAYS_COUNT];
    float dogs_rotations[DOGS_COUNT];
};
