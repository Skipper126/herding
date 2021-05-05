import time

import pygame

from herding import Herding
from herding.data.configuration import get_default_configuration
import argparse
import sys
import numpy as np
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    KEYUP,
    QUIT,
)


class ManualSteering:

    def __init__(self, env):
        self.env = env
        self.player_input = [1, 1]
        self.other_dogs_input = ([1, 1],) * (env.env_data.config.dogs_count - 1)
        self.quit = False

    def key_press(self, event):
        if event.key == K_UP:
            self.player_input[0] = 2
        elif event.key == K_DOWN:
            self.player_input[0] = 0
        elif event.key == K_LEFT:
            self.player_input[1] = 0
        elif event.key == K_RIGHT:
            self.player_input[1] = 2
        elif event.key == K_ESCAPE:
            self.quit = True

    def key_release(self, event):
        if event.key == K_UP:
            self.player_input[0] = 1
        elif event.key == K_DOWN:
            self.player_input[0] = 1
        elif event.key == K_LEFT:
            self.player_input[1] = 1
        elif event.key == K_RIGHT:
            self.player_input[1] = 1

    def process_events(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                self.key_press(event)
            elif event.type == KEYUP:
                self.key_release(event)
            elif event.type == QUIT:
                self.quit = True

    def run_env(self):
        self.env.reset()
        self.env.render()
        episode_reward = 0

        while not self.quit:
            self.process_events()

            env_input = np.array((self.player_input,) + self.other_dogs_input)
            _, reward, terminal, _ = self.env.step(env_input)
            episode_reward += reward
            self.env.render()
            if terminal:
                print(f'episode reward: {episode_reward}')
                self.env.reset()
                episode_reward = 0

        self.env.close()

    @staticmethod
    def print_debug(*args):
        print('\r', end='', flush=True)
        for arg in args:
            print(str(arg) + '\t', end='', flush=True)


def play(my_env=None):
    env = my_env or Herding()
    manual_steering = ManualSteering(env)
    manual_steering.run_env()


if __name__ == '__main__':
    args = {'sheep_count' : 24, 'dogs_count': 1}
    if len(sys.argv) > 1:
        config = get_default_configuration()
        parser = argparse.ArgumentParser()
        for name, value in config.items():
            parser.add_argument('--' + name, type=type(value))
        args_all = vars(parser.parse_args())
        for name, value in args_all.items():
            if args_all[name] is not None:
                args[name] = value

    play(Herding(args))
