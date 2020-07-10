import ray
from ray import tune
from utils.rllib_multiagent_adapter import MultiAgentHerding

ray.init()

env = MultiAgentHerding()
obs_space = env.env.observation_space
act_space = env.env.action_space

stop = {
    "training_iteration": 2000
}

results = tune.run(
    "PG",
    stop=stop,
    config={
        "horizon": 2000,
        "monitor": True,
        "env": MultiAgentHerding,
        "multiagent": {
            "policies": {
                "pg_policy": (None, obs_space, act_space, {"gamma": 0.99})
            },
            "policy_mapping_fn": lambda agent_id: "pg_policy",
        }
    },
)

ray.shutdown()
