from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import sys
from typing import Callable
import random
import ray
from ray import tune
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marllib import marl

logger = logging.getLogger(__name__)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class Checkpoint:
    def __init__(self, env_name: str, map_name: str, trainer: Trainer, pmap: Callable):
        self.env_name = env_name
        self.map_name = map_name
        self.trainer = trainer
        self.pmap = pmap


class NullLogger:
    """Logger for RLlib to disable logging"""

    def __init__(self, config=None):
        self.config = config
        self.logdir = ""

    def _init(self):
        pass

    def on_result(self, result):
        pass

    def update_config(self, config):
        pass

    def close(self):
        pass

    def flush(self):
        pass


def find_key(dictionary: dict, target_key: str):
    if target_key in dictionary:
        return dictionary[target_key]

    for key, value in dictionary.items():
        if isinstance(value, dict):
            result = find_key(value, target_key)
            if result is not None:
                return result

    return None


def form_algo_dict() -> dict[str, tuple[str, Trainer]]:
    trainers_dict = {}

    core_path = os.path.join(os.path.dirname(marl.__file__), "algos/core")
    for algo_type in os.listdir(core_path):
        if not os.path.isdir(os.path.join(core_path, algo_type)):
            continue
        for algo in os.listdir(os.path.join(core_path, algo_type)):
            if algo.endswith(".py") and not algo.startswith("__"):
                module_name = algo[:-3]  # remove .py extension
                module_path = f"marllib.marl.algos.core.{algo_type}.{module_name}"
                module = importlib.import_module(module_path)

                trainer_class_name = module_name.upper() + "Trainer"
                trainer_class = getattr(module, trainer_class_name, None)
                if trainer_class is None:
                    for name, obj in inspect.getmembers(module):
                        if name.endswith("Trainer"):
                            trainers_dict[module_name] = obj
                else:
                    trainers_dict[module_name] = (algo_type, trainer_class)

    return trainers_dict


def update_config(config: dict):
    # Extract config
    env_name = config["env"].split("_")[0]
    map_name = config["env"][len(env_name) + 1 :]
    model_name = find_key(config, "custom_model")
    model_arch_args = find_key(config, "model_arch_args")
    algo_name = find_key(config, "algorithm")
    share_policy = find_key(config, "share_policy")
    agent_level_batch_update = find_key(config, "agent_level_batch_update")

    ######################
    ### environment info ###
    ######################
    env = marl.make_env(env_name, map_name)
    env_instance, env_info = env
    if algo_name == "ippo":
        algo_name = "ppo"
    elif algo_name == "idqn":
        algo_name = "dqn"
    elif algo_name == "ia2c":
        algo_name = "a2c"
    algorithm = dotdict({"name": algo_name, "algo_type": ALGO_DICT[algo_name][0]})
    model_instance, model_info = marl.build_model(env, algorithm, model_arch_args)
    ModelCatalog.register_custom_model(model_name, model_instance)

    env_info = env_instance.get_env_info()
    policy_mapping_info = env_info["policy_mapping_info"]
    agent_name_ls = env_instance.agents
    env_info["agent_name_ls"] = agent_name_ls
    env_instance.close()

    config["model"]["custom_model_config"].update(env_info)

    ######################
    ### policy sharing ###
    ######################

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    # whether to agent level batch update when shared model parameter:
    # True -> default_policy | False -> shared_policy
    shared_policy_name = (
        "default_policy" if agent_level_batch_update else "shared_policy"
    )
    if share_policy == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError(
                "in {}, policy can not be shared, change it to 1. group 2. individual".format(
                    map_name
                )
            )

        policies = {shared_policy_name}
        policy_mapping_fn = lambda agent_id, episode, **kwargs: shared_policy_name

    elif share_policy == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(
                        map_name
                    )
                )

            policies = {shared_policy_name}
            policy_mapping_fn = lambda agent_id, episode, **kwargs: shared_policy_name

        else:
            policies = {
                "policy_{}".format(i): (
                    None,
                    env_info["space_obs"],
                    env_info["space_act"],
                    {},
                )
                for i in groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0])
            )

    elif share_policy == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError(
                "in {}, agent number too large, we disable no sharing function".format(
                    map_name
                )
            )

        policies = {
            "policy_{}".format(i): (
                None,
                env_info["space_obs"],
                env_info["space_act"],
                {},
            )
            for i in range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)]
        )

    else:
        raise ValueError("wrong share_policy {}".format(share_policy))

    # if happo or hatrpo, force individual
    if algo_name in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError(
                "in {}, agent number too large, we disable no sharing function".format(
                    map_name
                )
            )

        policies = {
            "policy_{}".format(i): (
                None,
                env_info["space_obs"],
                env_info["space_act"],
                {},
            )
            for i in range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)]
        )

    config.update(
        {
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
        }
    )


def load_model(model_config: dict) -> Checkpoint:
    """load model from given path

    Args:
        model_config (dict): model config dict, containing "algo", "params_path" and "model_path"

    Returns:
        ckpt (Checkpoint): The checkpoint loaded
    """

    try:
        with open(model_config["params_path"], "r") as f:
            params = json.load(f)
    except Exception as e:
        logger.error("Error loading params: %s" % e)
        raise e

    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            configure_logging=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
        )

    update_config(params)
    params["seed"] = random.randint(0, 10000)
    algo = model_config.get("algo", find_key(params, "algorithm"))
    if algo == "ippo":
        algo = "ppo"
    elif algo == "idqn":
        algo = "dqn"
    elif algo == "ia2c":
        algo = "a2c"
    trainer = ALGO_DICT[algo][1](
        params, logger_creator=lambda config: NullLogger(config)
    )
    trainer.restore(model_config["model_path"])

    # This function (policy_map_fn) takes in actor_id (str), episode (int), returns the policy_id (str)
    # Most of the time, episode can be just 1
    pmap = find_key(trainer.config, "policy_mapping_fn")

    env_name = params["env"].split("_")[0]
    map_name = params["env"][len(env_name) + 1 :]

    return Checkpoint(env_name, map_name, trainer, pmap)


ALGO_DICT = form_algo_dict()


if __name__ == "__main__":
    default_model_path = os.path.join(
        os.path.dirname(__file__), "best_model/checkpoint"
    )
    default_params_path = os.path.join(
        os.path.dirname(__file__), "best_model/params.json"
    )

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, default=default_model_path)
    argparser.add_argument("--params_path", type=str, default=default_params_path)
    argparser.add_argument("--epoch", type=int, default=100)
    argparser.add_argument("--collect_data", action="store_true")
    argparser.add_argument("--record", action="store_true")
    args = argparser.parse_args()

    ckpt = load_model(
        {
            "model_path": args.model_path,
            "params_path": args.params_path,
        }
    )
    agent, pmap = ckpt.trainer, ckpt.pmap

    # prepare env
    env = marl.make_env(environment_name=ckpt.env_name, map_name=ckpt.map_name)
    env_instance, env_info = env
    if args.collect_data and ckpt.env_name in ["macad", "macarla"]:
        from macarla_gym.misc.experiment import DataCollectWrapper

        env_instance.env = DataCollectWrapper(env_instance.env)
    if args.record and ckpt.env_name in ["macad", "macarla"]:
        env_instance.env.env_config["record"] = True

    # Inference
    for _ in range(args.epoch):
        obs = env_instance.reset()
        done = {"__all__": False}
        states = {
            actor_id: agent.get_policy(pmap(actor_id, 1)).get_initial_state()
            for actor_id in obs
        }

        while not done["__all__"]:
            action_dict = {}
            for agent_id in obs.keys():
                (
                    action_dict[agent_id],
                    states[agent_id],
                    _,
                ) = agent.compute_single_action(
                    obs[agent_id],
                    states[agent_id],
                    policy_id=pmap(agent_id, 1),
                    explore=True,
                )

            obs, reward, done, info = env_instance.step(action_dict)

    env_instance.close()
    ray.shutdown()
    logger.info("Inference finished!")
