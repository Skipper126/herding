from herding.envs.assets.configuration.names import ConfigName as cn
import multiprocessing as mp
import math


def get_workers_ranges(agents_count, workers_count):
    workers_ranges = []

    j = 0
    for i in range(workers_count):
        workers_ranges.append([])
        for _ in range(int(math.ceil(agents_count / workers_count))):
            if j < agents_count:
                workers_ranges[i].append(j)
                j += 1
            else:
                break

    return workers_ranges


def get_workers_count(env_data):
    workers_count_config = env_data.config[cn.MAX_THREADS_COUNT]
    workers_count_agents = max(env_data.config[cn.DOGS_COUNT], env_data.config[cn.SHEEP_COUNT])

    workers_count = workers_count_config if workers_count_config > 0 else mp.cpu_count()

    if workers_count_agents < workers_count:
        workers_count = workers_count_agents

    return workers_count
