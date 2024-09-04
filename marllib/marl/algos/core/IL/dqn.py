"""IDQN policy class and trainer class definitions.
"""

from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_CONFIG, DQNTrainer

###########
### DQN ###
###########


IDQNTorchPolicy = DQNTorchPolicy.with_updates(
    name="IDQNTorchPolicy",
    get_default_config=lambda: DQN_CONFIG,
)


def get_policy_class_dqn(config_):
    if config_["framework"] == "torch":
        return IDQNTorchPolicy


IDQNTrainer = DQNTrainer.with_updates(
    name="IDQNTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_dqn,
)
