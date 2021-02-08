import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import CLIReporter

from rl.model import HerdingModel
from rl.herding_env_wrapper import HerdingEnvWrapper


ModelCatalog.register_custom_model("herding_model", HerdingModel)

env = HerdingEnvWrapper()
obs_space = env.observation_space
act_space = env.action_space

config = {
    "env": HerdingEnvWrapper,
    "model": {
        "custom_model": "herding_model"
    },
    "multiagent": {
        "policies": {
            "policy": (None, obs_space, act_space, {})
        },
        "policy_mapping_fn": lambda agent_id: "policy",
    },
    #"replay_sequence_length": 500,
    "horizon": 300,
    "num_gpus": 1,
    "num_workers": 7,
    "num_envs_per_worker": 1,
    "batch_mode": "complete_episodes"
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
        progress_reporter=CLIReporter(max_report_frequency=60)
        #resume=True
    )

if __name__ == "__main__":
    ray.init()

    run_learning({
        'dogs_count': 1,
        'sheep_count': 3,
    })

    ray.shutdown()
