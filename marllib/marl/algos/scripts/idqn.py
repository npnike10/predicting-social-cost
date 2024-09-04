"""
This script runs the Independent DQN algorithm.
"""

import json
from typing import Any, Dict
from ray.tune.analysis import ExperimentAnalysis
from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.core.IL.dqn import IDQNTrainer
from marllib.marl.algos.scripts.coma import restore_model


def run_idqn(
    model: Any, exp: Dict, run: Dict, env: Dict, stop: Dict, restore: Dict
) -> ExperimentAnalysis:
    """This script runs the Independent DQN algorithm using Ray RLlib.
    Args:
        :params model (str): The name of the model class to register.
        :params exp (dict): A dictionary containing all the learning settings.
        :params run (dict): A dictionary containing all the environment-related settings.
        :params env (dict): A dictionary specifying the condition for stopping the training.
        :params restore (bool): A flag indicating whether to restore training/rendering or not.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """

    ModelCatalog.register_custom_model("Base_Model", model)

    _param = AlgVar(exp)

    episode_limit = env["episode_limit"]
    train_batch_size = _param["train_batch_size"]
    lr = _param["lr"]
    grad_clip = _param["grad_clip"]
    dueling = _param["dueling"]
    double_q = _param["double_q"]
    prioritized_replay = _param["prioritized_replay"]
    n_step = _param["n_step"]
    buffer_size = _param["buffer_size"]
    target_network_update_frequency = _param["target_network_update_freq"]
    initial_epsilon = _param["initial_epsilon"]
    final_epsilon = _param["final_epsilon"]
    epsilon_timesteps = _param["epsilon_timesteps"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    config = {
        "train_batch_size": train_batch_size,
        "buffer_size": buffer_size,
        "grad_clip": grad_clip,
        "dueling": dueling,
        "double_q": double_q,
        "prioritized_replay": prioritized_replay,
        "n_step": n_step,
        "rollout_fragment_length": 1,
        "target_network_update_freq": target_network_update_frequency,
        "lr": lr if restore is None else 1e-10,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": initial_epsilon,
            "final_epsilon": final_epsilon,
            "epsilon_timesteps": epsilon_timesteps,
        },
        "model": {
            "custom_model": "Base_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": back_up_config,
        },
    }

    config.update(run)

    algorithm = exp["algorithm"]
    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    RUNNING_NAME = "_".join([algorithm, arch, map_name])
    model_path = restore_model(restore, exp)

    results = tune.run(
        IDQNTrainer,
        name=RUNNING_NAME,
        checkpoint_at_end=exp["checkpoint_end"],
        checkpoint_freq=exp["checkpoint_freq"],
        restore=model_path,
        stop=stop,
        config=config,
        verbose=1,
        progress_reporter=CLIReporter(),
        local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"],
    )

    return results
