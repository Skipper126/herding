import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from rl.model import HerdingModel
from rl.rllib_multiagent_adapter import MultiAgentHerding


ModelCatalog.register_custom_model("herding_model", HerdingModel)

env = MultiAgentHerding()
obs_space = env.env.observation_space
act_space = env.env.action_space

config = {
    "env": MultiAgentHerding,
    "model": {
        "custom_model": "herding_model"
    },
    "multiagent": {
        "policies": {
            "policy": (None, obs_space, act_space, {})
        },
        "policy_mapping_fn": lambda agent_id: "policy",
    },
    "replay_sequence_length": 5,
    "horizon": 5000,
    "num_gpus": 1,
    "num_workers": 6,
    "num_envs_per_worker": 4,
    "lr": 0.001,
}

def run_learning(env_config):
    config['env_config'] = env_config
    stop = {
        #"episodes_total": 10
    }
    tune.run(
        "PPO",
        name="Herding",
        stop=stop,
        config=config,
        trial_name_creator=lambda trial: 'Herding',
        trial_dirname_creator=lambda trial: 'Herding',
        keep_checkpoints_num=1,
        checkpoint_freq=5,
        resume=True
    )

if __name__ == "__main__":
    ray.init()

    run_learning({
        'dogs_count': 1,
        'sheep_count': 3,
    })

    ray.shutdown()
