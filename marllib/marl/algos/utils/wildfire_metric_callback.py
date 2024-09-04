from typing import Dict, Optional, TYPE_CHECKING
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker


class WildfireCallbacks(DefaultCallbacks):
    """Custom callback to log the fraction of total trees which remain healthy at the end of an episode and agent episode returns. It logs the fraction of healthy trees overall, in each selfish region and the episode return of each agent."""

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs when an episode is done.

        Parameters
        ----------
        worker : RolloutWorker
            Reference to the current rollout worker.
        base_env : BaseEnv
            BaseEnv running the episode. The underlying env object can be gotten by calling base_env.get_unwrapped().
        policies : Dict[PolicyID, Policy]
            Mapping of policy id to policy objects. In single agent mode there will only be a single "default_policy".
        episode : MultiAgentEpisode
            Episode object which contains episode state. You can use the `episode.user_data` dict to store temporary data, and `episode.custom_metrics` to store custom metrics for the episode.
        env_index : Optional[int], optional
            Obsoleted: The ID of the environment, which the episode belongs to., by default None
        kwargs : dict
            Forward compatibility placeholder.
        """

        if self.legacy_callbacks.get("on_episode_end"):
            self.legacy_callbacks["on_episode_end"](
                {
                    "env": base_env,
                    "policy": policies,
                    "episode": episode,
                }
            )

        # whether algorithm is an IQL based method (e.g. IQL, IDQN). Used to determine how to get agent's policy id
        alg_iql = False
        env = worker.env.env

        # log overall fraction of healthy trees
        episode.custom_metrics["fraction_healthy_trees"] = (
            1 - (env.burnt_trees + env.trees_on_fire) / env.grid_size_without_walls**2
        )

        if env.log_selfish_region_metrics:
            # loop over all agents
            for a in env.agents:
                # log fraction of healthy trees in selfish region of the agent
                episode.custom_metrics[
                    f"selfish_region_{a.index}_fraction_healthy_trees"
                ] = (
                    1
                    - (
                        env.selfish_region_burnt_trees[a.index]
                        + env.selfish_region_trees_on_fire[a.index]
                    )
                    / env.selfish_region_size[a.index]
                )

                # get agent's policy id
                if env.cooperative_reward:
                    if alg_iql:
                        agent_policy_id = episode.policy_mapping_fn(
                            f"{a.index}", episode, worker
                        )
                    else:
                        agent_policy_id = episode.policy_mapping_fn(
                            f"{a.index}", episode
                        )
                else:
                    if alg_iql:
                        agent_policy_id = episode.policy_mapping_fn(
                            f"{a.index}", worker
                        )
                    else:
                        if env.num_agents == 1:
                            agent_policy_id = episode.policy_mapping_fn(
                                f"{a.index}", episode
                            )
                        else:
                            agent_policy_id = episode.policy_mapping_fn(f"{a.index}")

                # log episode return of the agent
                episode.custom_metrics[f"agent{a.index}_episode_reward"] = (
                    episode.agent_rewards[(f"{a.index}", agent_policy_id)]
                )
