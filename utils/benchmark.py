from typing import NamedTuple
from herding import Herding
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time

class BenchmarkParams(NamedTuple):
    sheep_min = 1
    sheep_max = 1000
    sheep_step = 10
    dogs_min = 1
    dogs_max = 8
    dogs_step = 2
    single_benchmark_max_time = 10
    benchmark_iterations = 1000

def run_benchmark(iterations, **env_params):
    env = Herding(**env_params)
    dogs_count = env.env_data.config.dogs_count
    env.reset()
    start_time = time.time()
    for i in range(iterations):
        env.step(np.random.rand(dogs_count, 3).astype(np.float32))
    end_time = time.time()
    result = end_time - start_time
    print(str(int(iterations / result)) + " iterations / s")


def run_benchmarks(params: BenchmarkParams = BenchmarkParams(), log_dir:str = None, show_plot=False):
    dogs_counts = list(range(params.dogs_min, params.dogs_max, params.dogs_step))
    sheep_counts = list(range(params.sheep_min, params.sheep_max, params.sheep_step))
    actions = _get_actions(dogs_counts)
    results = np.zeros((params.dogs_max, params.sheep_max))


    for dogs_count in dogs_counts:
        for sheep_count in sheep_counts:
            env = Herding(dogs_count=dogs_count, sheep_count=sheep_count)
            env.reset()
            start_time = time.time()
            for i in range(params.benchmark_iterations):
                env.step(np.random.rand(dogs_count, 3))
            end_time = time.time()
            result = end_time - start_time
            results[dogs_count, sheep_count] = result

    if log_dir is not None:
        _write_log(results, log_dir)
    if show_plot:
        _show_plot(dogs_counts, sheep_counts, results)


def _write_log(results, log_dir):
    pass

def _show_plot(dogs_counts, sheep_counts, results):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(np.array(dogs_counts), np.array(sheep_counts))
    Z = results[X, Y]
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def _get_actions(dogs_counts):
    actions = []
    for dogs_count in dogs_counts:
        actions.append(np.empty((dogs_count, 3)))

    return actions
