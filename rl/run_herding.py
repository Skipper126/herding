from threading import local

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
import numpy as np
from herding import Herding
from rl.model import HerdingModel
from rl.herding_env_wrapper import HerdingEnvWrapper
ModelCatalog.register_custom_model("herding_model", HerdingModel)

env = HerdingEnvWrapper()
obs_space = env.observation_space
act_space = env.action_space
config = {
    "env": HerdingEnvWrapper,
    "env_config": {
    },
    "model": {
        "custom_model": "herding_model"
    },
    "multiagent": {
        "policies": {
            "policy": (None, obs_space, act_space, {})
        },
        "policy_mapping_fn": lambda agent_id: "policy",
    },
    "horizon": 2000,
    "num_gpus": 1,
    "explore": False
    #"replay_sequence_length": 5,
    #"num_workers": 4,
    #"num_envs_per_worker": 2,
}

ray.init(local_mode=True)

checkpoint_number = 790

env = Herding({
        "sheep_count": 3
        #"agents_layout": "simple"
    })
agent = ppo.PPOTrainer(config=config, env=HerdingEnvWrapper)
agent.restore(rf"C:\Users\Mateusz\ray_results\Herding\Herding\checkpoint_{checkpoint_number}\checkpoint-{checkpoint_number}")

while True:
    episode_reward = 0
    done = False
    steps = 0
    obs = env.reset()
    while (not done) and (steps != 300):
        action = agent.compute_action(obs[0], policy_id="policy")
        obs, reward, done, info = env.step(np.array([[2, action]]))
        env.render()
        episode_reward += reward
        steps += 1

ray.shutdown()
