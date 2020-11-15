from herding import Herding
from herding.data.configuration import get_default_configuration
from pyglet.window import key
import argparse
import sys


class ManualSteering:

    def __init__(self, env):
        self.env = env
        self.player_input = [0, 0]
        self.other_dogs_input = ([0, 0],) * (env.env_data.config.dogs_count - 1)
        self.quit = False

    def key_press(self, k, mod):
        if k == key.LEFT:
            self.player_input[0] = -1
        elif k == key.RIGHT:
            self.player_input[0] = 1
        elif k == key.UP:
            self.player_input[1] = 1
        elif k == key.DOWN:
            self.player_input[1] = -1
        elif k == key.ESCAPE:
            self.quit = True

    def key_release(self, k, mod):
        if k == key.LEFT:
            self.player_input[0] = 0
        elif k == key.RIGHT:
            self.player_input[0] = 0
        elif k == key.UP:
            self.player_input[1] = 0
        elif k == key.DOWN:
            self.player_input[1] = 0

    def on_close(self):
        self.quit = True

    def run_env(self):
        self.env.reset()
        self.env.render()

        self.env.viewer.viewer.window.on_key_press = self.key_press
        self.env.viewer.viewer.window.on_key_release = self.key_release
        self.env.viewer.viewer.window.on_close = self.on_close

        episode_reward = 0

        while not self.quit:
            env_input = (self.player_input,) + self.other_dogs_input
            observation, reward, terminal, _ = self.env.step(env_input)
            episode_reward += reward
            self.env.render()

            if terminal:
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
    args = {}
    if len(sys.argv) > 1:
        config = get_default_configuration()
        parser = argparse.ArgumentParser()
        for name, value in config.items():
            parser.add_argument('--' + name, type=type(value))
        args_all = vars(parser.parse_args())
        for name, value in args_all.items():
            if args_all[name] is not None:
                args[name] = value

    play(Herding(**args))
