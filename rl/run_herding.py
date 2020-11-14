import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog

from herding import Herding
from rl.model import HerdingModel
from rl.rllib_multiagent_adapter import MultiAgentHerding
ModelCatalog.register_custom_model("herding_model", HerdingModel)

env = MultiAgentHerding()
obs_space = env.env.observation_space
act_space = env.env.action_space
config = {
    "env": MultiAgentHerding,
    "env_config": {
        "sheep_count": 1,
        "agents_layout": "simple"
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
    #"num_workers": 4,
    #"num_envs_per_worker": 2,
}

ray.init()

env = Herding(**{
        "sheep_count": 3
        #"agents_layout": "simple"
    })
agent = ppo.PPOTrainer(config=config, env=MultiAgentHerding)
agent.restore(r"C:\Users\Mateusz\ray_results\Herding\Herding_543a\checkpoint_125\checkpoint-125")

episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = agent.compute_action(obs[0], policy_id="policy")
    obs, reward, done, info = env.step([action])
    env.render()
    episode_reward += reward

ray.shutdown()
