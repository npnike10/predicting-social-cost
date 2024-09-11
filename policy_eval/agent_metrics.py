# pylint: disable=import-error, wrong-import-position
import os
import json
from itertools import count
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_agent_policies, process_observation, select_action
import torch
import torch._dynamo
import numpy as np
import gym
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm
from wildfire_environment.utils.misc import (
    get_initial_fire_coordinates,
)

torch._dynamo.config.suppress_errors = True


class AgentMetrics:
    """Compute heuristic metrics for agents in WildfireEnv."""

    def __init__(
        self,
        env,
        gamma,
        episodes_per_state,
        initial_state_identifiers,
        mmdp_policy,
        mg_policy,
        stochastic_policy,
        mmdp_model_path,
        mmdp_params_path,
        mg_model_path,
        mg_params_path,
    ):
        """Create AgentMetrics object.

        Parameters
        ----------
        env : WildfireEnv
            WildfireEnv object
        gamma : float
            discount factor
        episodes_per_state : int
            number of episodes to run for each initial state
        initial_state_identifiers : list[tuple]
            list of tuples specifying the initial states over which to average the state visitation frequencies. Each tuple is an initial state identifier. It is the position of the center cell of the fire square, if it is odd sized. If the fire square is even sized, the top-left corner cell is the initial state identifier.
        mmdp_policy : str
            specifies the MMDP policy for which to generate heatmaps. It is the name of the corresponding training run.
        mg_policy : str
            specifies the MG policy for which to generate heatmaps. It is the name of the corresponding training run.
        stochastic_policy : bool
            whether policy is stochastic
        mmdp_model_path : str
            path to the file containing MMDP agent policies' models
        mmdp_params_path : str
            path to the params file for the MMDP policy training run. The file contains a dictionary. The dictionary in file at mmdp_params_path should not contain the key "callbacks" and corresponding value. A copy of the original params file may be used for this purpose.
        mg_model_path : str
            path to the file containing MG agent policies' models
        mg_params_path : str
            path to the params file for the MG policy training run. The file contains a dictionary. The dictionary in file at mg_params_path should not contain the key "callbacks" and corresponding value. A copy of the original params file may be used for this purpose.
        """
        self.env = env
        self.gamma = gamma
        self.episodes_per_state = episodes_per_state
        self.initial_state_identifiers = initial_state_identifiers
        self.mmdp_policy = mmdp_policy
        self.mg_policy = mg_policy
        self.stochastic_policy = stochastic_policy

        self.run = (
            "".join(str(i) for i in list(sum(initial_state_identifiers, ())))
            + "_"
            + str(episodes_per_state)
            + "_run"
        )  # run name

        # load agent policies
        self.mmdp_agent_policies = load_agent_policies(
            mmdp_model_path, mmdp_params_path, True, env.num_agents
        )
        self.mg_agent_policies = load_agent_policies(
            mg_model_path, mg_params_path, False, env.num_agents
        )

        # choose device on which PyTorch tensors will be allocated
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # store experiment configuration and results
        self.exp_data = {
            "num_episodes_per_state": episodes_per_state,
            "run": self.run,
            "input_model_dir_mmdp": mmdp_model_path,
            "input_params_dir_mmdp": mmdp_params_path,
            "input_model_dir_mg": mg_model_path,
            "input_params_dir_mg": mg_params_path,
            "initial state identifiers": initial_state_identifiers,
        }

        # compile select_action function for faster execution at runtime
        self.select_action_opt = torch.compile(select_action, mode="reduce-overhead")

    def state_visitation_heatmaps(self, initial_fire_vertices, selfish_region_vertices):
        """Create heatmaps of state visitation frequencies for MG and MMDP agents.

        Parameters
        ----------
        initial_fire_vertices : list[tuple]
            vertices of polygon formed by initial state identifiers
        selfish_region_vertices : list[list[tuple]]
            vertices of agents' selfish regions, in order of increasing agent index
        """
        env = self.env
        device = self.device

        # create directories to store results
        RESULTS_PATH = (
            "policy_eval/results/visitation_maps"  # directory to store results
        )
        if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

        # create arrays to store final state visitation frequencies
        final_mg_heatmap = [
            np.zeros((env.grid_size, env.grid_size)) for _ in range(env.num_agents)
        ]
        final_mmdp_heatmap = [
            np.zeros((env.grid_size, env.grid_size)) for _ in range(env.num_agents)
        ]

        # loop over all initial fire locations. Initial states only differ in the location of fire, so initial state identifiers specify the location of initial fire.
        for location in self.initial_state_identifiers:
            # create arrays to store state visitation frequencies for current initial state
            mg_heatmap = [
                np.zeros((env.grid_size, env.grid_size)) for _ in range(env.num_agents)
            ]
            mmdp_heatmap = [
                np.zeros((env.grid_size, env.grid_size)) for _ in range(env.num_agents)
            ]

            # reset env to current initial state and get initial observations
            trees_on_fire = get_initial_fire_coordinates(
                *location,
                env.grid_size,
                env.initial_fire_size,
            )
            initial_state = torch.tensor(
                env.construct_state(trees_on_fire, env.agent_start_positions, 0),
                dtype=torch.float32,
            ).to(device)
            obs, _ = env.reset(state=initial_state.cpu().numpy())
            ma_obs = process_observation(obs, device, initial_state)

            # record initial positions in heatmaps
            for i, agent in enumerate(env.agents):
                mg_heatmap[i][agent.pos[1], agent.pos[0]] += 1
                mmdp_heatmap[i][agent.pos[1], agent.pos[0]] += 1

            # run episodes to compute mg agent state visitation frequencies
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MG Episodes for initial fire at {location}",
            ):
                for _ in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mg_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # record agent positions in heatmaps
                    for i, agent in enumerate(env.agents):
                        mg_heatmap[i][agent.pos[1], agent.pos[0]] += 1

                    # check if episode is done
                    if done:
                        break

                    # process next observation
                    next_state = torch.tensor(env.get_state(), dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset env to initial state for next episode
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)

            # run episodes to compute mmdp agent state visitation frequencies
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MMDP Episodes for fire at {location}",
            ):
                for _ in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mmdp_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # record agent positions in heatmaps
                    for i, agent in enumerate(env.agents):
                        mmdp_heatmap[i][agent.pos[1], agent.pos[0]] += 1

                    # check if episode is done
                    if done:
                        break

                    # process next observation
                    next_state = torch.tensor(env.get_state(), dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset env to initial state for next episode
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)

            # average out state visitation frequencies over all episodes for current initial state
            for hm in mg_heatmap:
                hm /= self.episodes_per_state
            for hm in mmdp_heatmap:
                hm /= self.episodes_per_state

            # record averaged state visitation frequencies for current initial state in final heatmaps
            for i, hm in enumerate(mg_heatmap):
                final_mg_heatmap[i] += hm
            for i, hm in enumerate(mmdp_heatmap):
                final_mmdp_heatmap[i] += hm

        # average out state visitation frequencies over all initial states
        for hm in final_mg_heatmap:
            hm /= len(self.initial_state_identifiers)
        for hm in final_mmdp_heatmap:
            hm /= len(self.initial_state_identifiers)

        # save experiment data
        with open(
            f"{RESULTS_PATH}/{self.run}_exp_data.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(self.exp_data, fp, sort_keys=True, indent=4)
        with open(
            f"{RESULTS_PATH}/{self.mmdp_policy}_{self.mg_policy}_policies_{self.run}_exp_data.json",
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(self.exp_data, fp, sort_keys=True, indent=4)

        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=480)

        # create mmdp heatmaps
        for i, hm in enumerate(final_mmdp_heatmap):
            ax = axes[0, i]
            cbar_location = ["left", "right"]
            heatmap_plot = sns.heatmap(
                hm,
                cmap="GnBu",
                vmin=np.min(hm),
                vmax=np.max(hm),
                ax=ax,
                cbar_kws=dict(use_gridspec=False, location=cbar_location[i]),
            )
            heatmap_plot.xaxis.tick_top()

            # create patches for initial fire, selfish regions, agent initial positions and walls
            initial_fire_patch = patches.Polygon(
                xy=initial_fire_vertices,
                closed=True,
                edgecolor="orange",
                facecolor="none",
                linewidth=1,
            )
            selfish_region_patch = patches.Polygon(
                xy=selfish_region_vertices[i],
                closed=True,
                edgecolor=f"{env.agent_colors[i]}",
                facecolor="none",
                linewidth=1,
            )
            upper_wall_patch = patches.Rectangle(
                xy=(0, 0),
                width=env.grid_size,
                height=1,
                edgecolor="gray",
                facecolor="gray",
            )
            lower_wall_patch = patches.Rectangle(
                xy=(0, env.grid_size - 1),
                width=env.grid_size,
                height=1,
                edgecolor="gray",
                facecolor="gray",
            )
            left_wall_patch = patches.Rectangle(
                xy=(0, 0),
                width=1,
                height=env.grid_size,
                edgecolor="gray",
                facecolor="gray",
            )
            right_wall_patch = patches.Rectangle(
                xy=(env.grid_size - 1, 0),
                width=1,
                height=env.grid_size,
                edgecolor="gray",
                facecolor="gray",
            )
            initial_position_patch = patches.Circle(
                xy=(
                    env.agent_start_positions[i][0] + 0.5,
                    env.agent_start_positions[i][1] + 0.5,
                ),
                radius=0.2,
                edgecolor="olive",
                facecolor="none",
                linewidth=7,
            )

            # add patches to heatmap plot
            ax.add_patch(initial_fire_patch)
            ax.add_patch(initial_position_patch)
            ax.add_patch(selfish_region_patch)
            ax.add_patch(upper_wall_patch)
            ax.add_patch(lower_wall_patch)
            ax.add_patch(left_wall_patch)
            ax.add_patch(right_wall_patch)

            # set plot title and save plot
            ax.set_title(
                f"Average State Visitation Frequencies of MMDP Agent {i}",
            )

        # create mg heatmaps
        for i, hm in enumerate(final_mg_heatmap):
            ax = axes[1, i]
            cbar_location = ["left", "right"]
            heatmap_plot = sns.heatmap(
                hm,
                cmap="GnBu",
                vmin=np.min(hm),
                vmax=np.max(hm),
                ax=ax,
                cbar_kws=dict(use_gridspec=False, location=cbar_location[i]),
            )
            heatmap_plot.xaxis.tick_top()

            # create patches for initial fire, selfish regions, agent initial positions and walls
            initial_fire_patch = patches.Polygon(
                xy=initial_fire_vertices,
                closed=True,
                edgecolor="orange",
                facecolor="none",
                linewidth=1,
            )
            selfish_region_patch = patches.Polygon(
                xy=selfish_region_vertices[i],
                closed=True,
                edgecolor=f"{env.agent_colors[i]}",
                facecolor="none",
                linewidth=1,
            )
            upper_wall_patch = patches.Rectangle(
                xy=(0, 0),
                width=env.grid_size,
                height=1,
                edgecolor="gray",
                facecolor="gray",
            )
            lower_wall_patch = patches.Rectangle(
                xy=(0, env.grid_size - 1),
                width=env.grid_size,
                height=1,
                edgecolor="gray",
                facecolor="gray",
            )
            left_wall_patch = patches.Rectangle(
                xy=(0, 0),
                width=1,
                height=env.grid_size,
                edgecolor="gray",
                facecolor="gray",
            )
            right_wall_patch = patches.Rectangle(
                xy=(env.grid_size - 1, 0),
                width=1,
                height=env.grid_size,
                edgecolor="gray",
                facecolor="gray",
            )
            initial_position_patch = patches.Circle(
                xy=(
                    env.agent_start_positions[i][0] + 0.5,
                    env.agent_start_positions[i][1] + 0.5,
                ),
                radius=0.2,
                edgecolor=f"{env.agent_colors[i]}",
                facecolor="none",
                linewidth=7,
            )

            # add patches to heatmap plot
            ax.add_patch(initial_fire_patch)
            ax.add_patch(initial_position_patch)
            ax.add_patch(selfish_region_patch)
            ax.add_patch(upper_wall_patch)
            ax.add_patch(lower_wall_patch)
            ax.add_patch(left_wall_patch)
            ax.add_patch(right_wall_patch)

            # set plot title and save plot
            ax.set_title(
                f"Average State Visitation Frequencies of MG Agent {i}",
            )

        # save plots
        plt.savefig(
            f"{RESULTS_PATH}/{self.mmdp_policy}_{self.mg_policy}_policies_{self.run}.png"
        )

    def distance_to_fire_boundary(self, agent_pos):
        """
        Returns the manhattan distance to, and position of, the closest tree in fire boundary from specified position.

        Parameters
        ----------
        agent_pos : tuple
            position of the agent

        Returns
        -------
        tuple
            manhattan distance to the closest tree in fire boundary and position of the tree. If there is no fire boundary, returns -1 and the agent position. If the agent is on the fire boundary, returns 0 and the agent position.
        """
        o = self.env.helper_grid.get(*agent_pos)
        if o is None or o.type == "tree":
            if o.state == 1 and self.is_on_fire_boundary(agent_pos):
                return 0, agent_pos

        min_dist = self.env.grid_size_without_walls * 10
        for o in self.env.helper_grid.grid:
            if (
                o is not None
                and o.type == "tree"
                and o.state == 1
                and self.is_on_fire_boundary(o.pos)
            ):
                dist = self.manhattan_distance(agent_pos, o.pos)
                if dist < min_dist:
                    min_dist = dist
                    tree_pos = o.pos
        if min_dist == self.env.grid_size_without_walls * 10:
            return -1, agent_pos
        return (
            min_dist,
            tree_pos,
        )

    def is_on_fire_boundary(self, pos):
        """
        Check if the specified position is in the fire boundary. A position is in the fire boundary if at least one neighbor is a healthy tree. Neighbors are in the four cardinal directions.

        Returns
        -------
        bool
            whether the specified position is in the fire boundary
        """
        on_boundary = False
        if (
            self.env.helper_grid.get(*pos) is not None
            and self.env.helper_grid.get(*pos).type == "tree"
            and self.env.helper_grid.get(*pos).state == 1
        ):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                o = self.env.helper_grid.get(pos[0] + dx, pos[1] + dy)
                if o is not None and o.type == "tree" and o.state == 0:
                    on_boundary = True
        return on_boundary

    def manhattan_distance(self, pos1, pos2):
        """
        Returns the manhattan distance between two positions.

        Parameters
        ----------
        pos1 : tuple
            a position on grid
        pos2 : tuple
            a position on grid

        Returns
        -------
        int
            manhattan distance between the two positions
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def boundary_attack_metric(self):
        """Compute the boundary attack metric for MG and MMDP agents. The boundary attack metric measures if the actions taken by the agent move it closer to the fire boundary."""

        env = self.env
        device = self.device

        # create directories to store results
        mmdp_results_path = f"policy_eval/results/{self.mmdp_policy}_policy/strategy_metrics/boundary_attack_metric"
        mg_results_path = f"policy_eval/results/{self.mg_policy}_policy/strategy_metrics/boundary_attack_metric"
        if not os.path.exists(mmdp_results_path):
            os.makedirs(mmdp_results_path)
        if not os.path.exists(mg_results_path):
            os.makedirs(mg_results_path)

        # initialize arrays to store final metric values
        final_mg_metric = np.zeros(env.num_agents)
        final_mmdp_metric = np.zeros(env.num_agents)

        # loop over all initial fire locations
        for loc in self.initial_state_identifiers:
            # initialize arrays to store metric values averaged across all episodes for current initial fire location
            mg_metric_current_fire = np.zeros(env.num_agents)
            mmdp_metric_current_fire = np.zeros(env.num_agents)

            # initialize arrays to store manhattan distances from fire boundary
            mg_dm = np.zeros(env.num_agents)
            mmdp_dm = np.zeros(env.num_agents)
            mg_dm_next = np.zeros(env.num_agents)
            mmdp_dm_next = np.zeros(env.num_agents)

            # reset env to current initial state and get initial observations
            trees_on_fire = get_initial_fire_coordinates(
                *loc,
                env.grid_size,
                env.initial_fire_size,
            )
            initial_state = torch.tensor(
                env.construct_state(trees_on_fire, env.agent_start_positions, 0),
                dtype=torch.float32,
            ).to(device)
            obs, _ = env.reset(state=initial_state.cpu().numpy())
            ma_obs = process_observation(obs, device, initial_state)

            # calculate manhattan distance from closest tree in fire boundary at episode start
            for i, agent in enumerate(env.agents):
                initial_dm, _ = self.distance_to_fire_boundary(agent.pos)
                mg_dm[i] = initial_dm
                mmdp_dm[i] = initial_dm

            # loop to run episodes for mg agent
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MG Episodes for fire center at {loc}",
            ):
                # initialize arrays to store metric value for current episode
                mg_metric_ep = np.zeros(env.num_agents)
                mmdp_metric_ep = np.zeros(env.num_agents)
                for t in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mg_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # check if episode is done
                    if done:
                        # normalize metric
                        mg_metric_ep *= (1 - self.gamma) / (1 - self.gamma ** (t + 1))
                        break

                    # calculate manhattan distance from fire boundary
                    for i, agent in enumerate(env.agents):
                        dm_next, _ = self.distance_to_fire_boundary(agent.pos)
                        # break if there is no fire boundary
                        if dm_next == -1:
                            # normalize metric
                            mg_metric_ep *= (1 - self.gamma) / (
                                1 - self.gamma ** (t + 1)
                            )
                            break
                        else:
                            mg_dm_next[i] = dm_next

                    # calculate metric
                    for i, agent in enumerate(env.agents):
                        if mg_dm[i] == 0 and mg_dm_next[i] == 0:
                            mg_metric_ep[i] += self.gamma**t
                        else:
                            mg_metric_ep[i] += self.gamma**t * np.heaviside(
                                mg_dm[i] - mg_dm_next[i], 0
                            )

                    # update mg_dm for next step
                    for i, agent in enumerate(env.agents):
                        mg_dm[i] = mg_dm_next[i]

                    # process next observation
                    next_state = env.get_state()
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset environment to initial state
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)
                # update mg_metric_current_fire with metric for current episode
                mg_metric_current_fire += mg_metric_ep
                # reset dm for next episode
                for i, agent in enumerate(env.agents):
                    mg_dm[i] = initial_dm

            # loop to run episodes for mmdp agent
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MMDP Episodes for fire center at {loc}",
            ):
                for t in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mmdp_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # check if episode is done
                    if done:
                        # normalize metric
                        mmdp_metric_ep *= (1 - self.gamma) / (1 - self.gamma ** (t + 1))
                        break

                    # calculate manhattan distance from fire boundary
                    for i, agent in enumerate(env.agents):
                        dm_next, _ = self.distance_to_fire_boundary(agent.pos)
                        # break if there is no fire boundary
                        if dm_next == -1:
                            # normalize metric
                            mmdp_metric_ep *= (1 - self.gamma) / (
                                1 - self.gamma ** (t + 1)
                            )
                            break
                        else:
                            mmdp_dm_next[i] = dm_next

                    # calculate metric
                    for i, agent in enumerate(env.agents):
                        if mmdp_dm[i] == 0 and mmdp_dm_next[i] == 0:
                            mmdp_metric_ep[i] += self.gamma**t
                        else:
                            mmdp_metric_ep[i] += self.gamma**t * np.heaviside(
                                mmdp_dm[i] - mmdp_dm_next[i], 0
                            )

                    # update mmdp_dm for next step
                    for i, agent in enumerate(env.agents):
                        mmdp_dm[i] = mmdp_dm_next[i]

                    # process next observation
                    next_state = env.get_state()
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset environment to initial state
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)
                # update mmdp_metric_current_fire with metric for current episode
                mmdp_metric_current_fire += mmdp_metric_ep
                # reset dm for next episode
                for i, agent in enumerate(env.agents):
                    mmdp_dm[i] = initial_dm

            # average out metric values over all episodes
            mg_metric_current_fire /= self.episodes_per_state
            mmdp_metric_current_fire /= self.episodes_per_state

            # update final metric values with metric values for current initial fire location
            final_mg_metric += mg_metric_current_fire
            final_mmdp_metric += mmdp_metric_current_fire

        # average out final metric values over all initial fire locations
        final_mg_metric /= len(self.initial_state_identifiers)
        final_mmdp_metric /= len(self.initial_state_identifiers)

        # save exp config
        exp_data = self.exp_data.copy()
        exp_data["final_mg_metric"] = final_mg_metric.tolist()
        exp_data["final_mmdp_metric"] = final_mmdp_metric.tolist()
        with open(
            f"{mmdp_results_path}/{self.run}_exp_config.json",
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(exp_data, fp, sort_keys=True, indent=4)
        with open(
            f"{mg_results_path}/{self.run}_exp_config.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(exp_data, fp, sort_keys=True, indent=4)
        return exp_data

    def distance_from_other_agents_metric(self):
        """For each MG and MMDP agent, compute metric measuring the normalized (by maximum possible distance) average manhattan distance from all other agents during an episode."""

        env = self.env
        device = self.device

        # directories to store results
        mmdp_results_path = f"policy_eval/results/{self.mmdp_policy}_policy/strategy_metrics/distance_from_other_agents"
        mg_results_path = f"policy_eval/results/{self.mg_policy}_policy/strategy_metrics/distance_from_other_agents"
        if not os.path.exists(mmdp_results_path):
            os.makedirs(mmdp_results_path)
        if not os.path.exists(mg_results_path):
            os.makedirs(mg_results_path)

        # initialize arrays to store final metric values
        final_mg_metric = np.zeros(env.num_agents)
        final_mmdp_metric = np.zeros(env.num_agents)

        # loop over all initial fire locations
        for loc in self.initial_state_identifiers:
            # initialize arrays to store metric values averaged across all episodes for current initial fire location
            mg_metric_current_fire = np.zeros(env.num_agents)
            mmdp_metric_current_fire = np.zeros(env.num_agents)

            # reset env to current initial state and get initial observations
            trees_on_fire = get_initial_fire_coordinates(
                *loc,
                env.grid_size,
                env.initial_fire_size,
            )
            initial_state = torch.tensor(
                env.construct_state(trees_on_fire, env.agent_start_positions, 0),
                dtype=torch.float32,
            ).to(device)
            obs, _ = env.reset(state=initial_state.cpu().numpy())
            ma_obs = process_observation(obs, device, initial_state)

            # loop to run episodes for mg agent
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MG Episodes for fire center at {loc}",
            ):
                # initialize array to store metric value for current episode
                mg_metric_ep = np.zeros(env.num_agents)

                # update mg_metric_ep with mean distance from other agents at episode start
                for i, agent in enumerate(env.agents):
                    distances = [
                        self.manhattan_distance(agent.pos, other_agent.pos)
                        for other_agent in env.agents
                        if agent != other_agent
                    ]
                    mg_metric_ep[i] += np.mean(distances)

                for t in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mg_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # update mg_metric_ep with mean distance from other agents at current step
                    for i, agent in enumerate(env.agents):
                        distances = [
                            self.manhattan_distance(agent.pos, other_agent.pos)
                            for other_agent in env.agents
                            if agent != other_agent
                        ]
                        mg_metric_ep[i] += np.mean(distances)

                    # check if episode is done
                    if done:
                        for i in range(env.num_agents):
                            # average out metric over number of steps in episode
                            mg_metric_ep[i] /= t + 1
                            # normalize metric
                            mg_metric_ep[i] /= 2 * (env.grid_size_without_walls - 1)
                        break

                    # process next observation
                    next_state = env.get_state()
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset environment to initial state
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)
                # update mg_metric_current_fire with metric for current episode
                mg_metric_current_fire += mg_metric_ep

            # loop to run episodes for mmdp agent
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MMDP Episodes for fire center at {loc}",
            ):
                # initialize array to store metric value for current episode
                mmdp_metric_ep = np.zeros(env.num_agents)

                # update mg_metric_ep with mean distance from other agents at episode start
                for i, agent in enumerate(env.agents):
                    distances = [
                        self.manhattan_distance(agent.pos, other_agent.pos)
                        for other_agent in env.agents
                        if agent != other_agent
                    ]
                    mmdp_metric_ep[i] += np.mean(distances)

                for t in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mmdp_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # update mmdp_metric_ep with mean distance from other agents at current step
                    for i, agent in enumerate(env.agents):
                        distances = [
                            self.manhattan_distance(agent.pos, other_agent.pos)
                            for other_agent in env.agents
                            if agent != other_agent
                        ]
                        mmdp_metric_ep[i] += np.mean(distances)

                    # check if episode is done
                    if done:
                        for i in range(env.num_agents):
                            # average out metric over number of steps in episode
                            mmdp_metric_ep[i] /= t + 1
                            # normalize metric
                            mmdp_metric_ep[i] /= 2 * (env.grid_size_without_walls - 1)
                        break

                    # process next observation
                    next_state = env.get_state()
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset environment to initial state
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)
                # update mmdp_metric_current_fire with metric for current episode
                mmdp_metric_current_fire += mmdp_metric_ep

            # average out metric values over all episodes
            mg_metric_current_fire /= self.episodes_per_state
            mmdp_metric_current_fire /= self.episodes_per_state

            # update final metric values with metric values for current initial fire location
            final_mg_metric += mg_metric_current_fire
            final_mmdp_metric += mmdp_metric_current_fire

        # average out final metric values over all initial states
        final_mg_metric /= len(self.initial_state_identifiers)
        final_mmdp_metric /= len(self.initial_state_identifiers)

        # save experiment data
        exp_data = self.exp_data.copy()
        exp_data["final_mg_metric"] = final_mg_metric.tolist()
        exp_data["final_mmdp_metric"] = final_mmdp_metric.tolist()
        with open(
            f"{mmdp_results_path}/{self.run}_exp_config.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(exp_data, fp, sort_keys=True, indent=4)
        with open(
            f"{mg_results_path}/{self.run}_exp_config.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(exp_data, fp, sort_keys=True, indent=4)

        return exp_data

    def distance_from_selfish_region(self, xmin, xmax, ymin, ymax, agent_pos):
        """
        Returns the manhattan distance of agent from closest cell inside selfish region.

        Parameters
        ----------
        xmin : int
            x-coordinate of the left boundary of the selfish region
        xmax : int
            x-coordinate of the right boundary of the selfish region
        ymin : int
            y-coordinate of the top boundary of the selfish region
        ymax : int
            y-coordinate of the bottom boundary of the selfish region
        agent_pos : tuple
            position of the agent

        Returns
        -------
        int
            manhattan distance of agent from closest cell inside selfish region
        """
        if agent_pos[0] < xmin:
            if agent_pos[1] < ymin:
                return self.manhattan_distance(agent_pos, (xmin, ymin))
            elif agent_pos[1] > ymax:
                return self.manhattan_distance(agent_pos, (xmin, ymax))
            else:
                return xmin - agent_pos[0]
        elif agent_pos[0] > xmax:
            if agent_pos[1] < ymin:
                return self.manhattan_distance(agent_pos, (xmax, ymin))
            elif agent_pos[1] > ymax:
                return self.manhattan_distance(agent_pos, (xmax, ymax))
            else:
                return agent_pos[0] - xmax
        else:
            if agent_pos[1] < ymin:
                return ymin - agent_pos[1]
            elif agent_pos[1] > ymax:
                return agent_pos[1] - ymax
            else:
                return 0

    def distance_from_selfish_region_metric(self):
        """For each MG and MMDP agent, compute metric measuring the normalized (by maximum possible distance) average manhattan distance from the closest cell inside the selfish region of the agent during an episode."""

        env = self.env
        device = self.device

        # directories to store results
        mmdp_results_path = f"policy_eval/results/{self.mmdp_policy}_policy/strategy_metrics/distance_from_selfish_region"
        mg_results_path = f"policy_eval/results/{self.mg_policy}_policy/strategy_metrics/distance_from_selfish_region"
        if not os.path.exists(mmdp_results_path):
            os.makedirs(mmdp_results_path)
        if not os.path.exists(mg_results_path):
            os.makedirs(mg_results_path)

        # initialize arrays to store final metric values
        final_mg_metric = np.zeros(env.num_agents)
        final_mmdp_metric = np.zeros(env.num_agents)

        # loop over all initial fire locations
        for loc in self.initial_state_identifiers:
            # initialize metric values for current initial fire location
            mg_metric_current_fire = np.zeros(env.num_agents)
            mmdp_metric_current_fire = np.zeros(env.num_agents)

            # reset env to current initial state and get initial observations
            trees_on_fire = get_initial_fire_coordinates(
                *loc,
                env.grid_size,
                env.initial_fire_size,
            )
            initial_state = torch.tensor(
                env.construct_state(trees_on_fire, env.agent_start_positions, 0),
                dtype=torch.float32,
            ).to(device)
            obs, _ = env.reset(state=initial_state.cpu().numpy())
            ma_obs = process_observation(obs, device, initial_state)

            # loop to run episodes for mg agent
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MG Episodes for fire center at {loc}",
            ):
                # initialize array to store metric value for current episode
                mg_distances_ep = np.zeros(env.num_agents)

                # update mg_distances_ep with distance from selfish region at episode start
                for i, agent in enumerate(env.agents):
                    mg_distances_ep[i] += self.distance_from_selfish_region(
                        env.selfish_xmin[i],
                        env.selfish_xmax[i],
                        env.selfish_ymin[i],
                        env.selfish_ymax[i],
                        agent.pos,
                    )

                for t in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mg_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # calculate metric
                    for i, agent in enumerate(env.agents):
                        mg_distances_ep[i] += self.distance_from_selfish_region(
                            env.selfish_xmin[i],
                            env.selfish_xmax[i],
                            env.selfish_ymin[i],
                            env.selfish_ymax[i],
                            agent.pos,
                        )

                    # check if episode is done
                    if done:
                        for i in range(env.num_agents):
                            # average our metric by over number of steps in episode
                            mg_distances_ep[i] /= t + 1
                            # normalize metric
                            mg_distances_ep[i] /= 2 * (env.grid_size_without_walls - 1)
                        break

                    # process next observation
                    next_state = env.get_state()
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset environment to initial state
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)
                # update mg_metric_current_fire with metric for current episode
                mg_metric_current_fire += mg_distances_ep

            # loop to run episodes for mmdp agent
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MMDP Episodes for fire center at {loc}",
            ):
                # initialize array to store metric value for current episode
                mmdp_distances_ep = np.zeros(env.num_agents)

                # update mmdp_distances_ep with distance from selfish region at episode start
                for i, agent in enumerate(env.agents):
                    mmdp_distances_ep[i] += self.distance_from_selfish_region(
                        env.selfish_xmin[i],
                        env.selfish_xmax[i],
                        env.selfish_ymin[i],
                        env.selfish_ymax[i],
                        agent.pos,
                    )

                for t in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mmdp_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # calculate metric
                    for i, agent in enumerate(env.agents):
                        mmdp_distances_ep[i] += self.distance_from_selfish_region(
                            env.selfish_xmin[i],
                            env.selfish_xmax[i],
                            env.selfish_ymin[i],
                            env.selfish_ymax[i],
                            agent.pos,
                        )

                    # check if episode is done
                    if done:
                        for i in range(env.num_agents):
                            # average out metric over number of steps in episode
                            mmdp_distances_ep[i] /= t + 1
                            # normalize metric
                            mmdp_distances_ep[i] /= 2 * (
                                env.grid_size_without_walls - 1
                            )
                        break

                    # process next observation
                    next_state = env.get_state()
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset environment to initial state
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)
                # update mmdp_metric_current_fire with metric for current episode
                mmdp_metric_current_fire += mmdp_distances_ep

            # average out metric values over all episodes
            mg_metric_current_fire /= self.episodes_per_state
            mmdp_metric_current_fire /= self.episodes_per_state

            # update final metric values with metric values for current initial fire location
            final_mg_metric += mg_metric_current_fire
            final_mmdp_metric += mmdp_metric_current_fire

        # average out final metric values over all initial states
        final_mg_metric /= len(self.initial_state_identifiers)
        final_mmdp_metric /= len(self.initial_state_identifiers)

        # save experiment data
        exp_data = self.exp_data.copy()
        exp_data["final_mg_metric"] = final_mg_metric.tolist()
        exp_data["final_mmdp_metric"] = final_mmdp_metric.tolist()
        with open(
            f"{mmdp_results_path}/{self.run}_exp_config.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(exp_data, fp, sort_keys=True, indent=4)
        with open(
            f"{mg_results_path}/{self.run}_exp_config.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(exp_data, fp, sort_keys=True, indent=4)

        return exp_data

    def time_over_fire_metric(self, in_selfish_region=False):
        """For each MG and MMDP agent, compute metric measuring the normalized (by episode length) average time spent over trees on fire during an episode.

        Parameters
        ----------
        in_selfish_region : bool, optional
            whether to consider only the time spent over trees on fire in the selfish region. By default False.
        """

        env = self.env
        device = self.device

        # directories to store results
        mmdp_results_path = (
            f"policy_eval/results/{self.mmdp_policy}_policy/strategy_metrics/time_over_fire"
            + in_selfish_region * "_in_selfish_region"
        )
        mg_results_path = (
            f"policy_eval/results/{self.mg_policy}_policy/strategy_metrics/time_over_fire"
            + in_selfish_region * "_in_selfish_region"
        )
        if not os.path.exists(mmdp_results_path):
            os.makedirs(mmdp_results_path)
        if not os.path.exists(mg_results_path):
            os.makedirs(mg_results_path)

        # initialize arrays to store final metric values
        final_mg_metric = np.zeros(env.num_agents)
        final_mmdp_metric = np.zeros(env.num_agents)

        # loop over all initial fire locations
        for loc in self.initial_state_identifiers:
            # initialize arrays to store metric values for current initial fire location
            mg_metric_current_fire = np.zeros(env.num_agents)
            mmdp_metric_current_fire = np.zeros(env.num_agents)

            # reset env to current initial state and get initial observations
            trees_on_fire = get_initial_fire_coordinates(
                *loc,
                env.grid_size,
                env.initial_fire_size,
            )
            initial_state = torch.tensor(
                env.construct_state(trees_on_fire, env.agent_start_positions, 0),
                dtype=torch.float32,
            ).to(device)
            obs, _ = env.reset(state=initial_state.cpu().numpy())
            ma_obs = process_observation(obs, device, initial_state)

            # loop to run episodes for mg agent
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MG Episodes for fire center at {loc}",
            ):
                # initialize metric for current episode
                mg_metric_ep = np.zeros(env.num_agents)

                # update metric at episode start
                for i, agent in enumerate(env.agents):
                    if env.helper_grid.get(*agent.pos).state == 1:
                        if in_selfish_region:
                            if env.in_selfish_region(*agent.pos, agent.index):
                                mg_metric_ep[i] += 1
                        else:
                            mg_metric_ep[i] += 1

                for t in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mg_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # update metric
                    for i, agent in enumerate(env.agents):
                        if env.helper_grid.get(*agent.pos).state == 1:
                            if in_selfish_region:
                                if env.in_selfish_region(*agent.pos, agent.index):
                                    mg_metric_ep[i] += 1
                            else:
                                mg_metric_ep[i] += 1

                    # check if episode is done
                    if done:
                        # normalize metric
                        mg_metric_ep /= t + 1
                        break

                    # process next observation
                    next_state = env.get_state()
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset environment to initial state
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)
                # update mg_metric_current_fire with metric for current episode
                mg_metric_current_fire += mg_metric_ep

            # loop to run episodes for mmdp agent
            for _ in tqdm(
                range(self.episodes_per_state),
                desc=f"MMDP Episodes for fire center at {loc}",
            ):
                # initialize metric for current episode
                mmdp_metric_ep = np.zeros(env.num_agents)

                # update metric at episode start
                for i, agent in enumerate(env.agents):
                    if env.helper_grid.get(*agent.pos).state == 1:
                        if in_selfish_region:
                            if env.in_selfish_region(*agent.pos, agent.index):
                                mmdp_metric_ep[i] += 1
                        else:
                            mmdp_metric_ep[i] += 1

                for t in count():
                    # step environment
                    action = self.select_action_opt(
                        ma_obs,
                        self.mmdp_agent_policies,
                        env.num_agents,
                        self.stochastic_policy,
                    )
                    next_obs, _, done, _ = env.step(action)

                    # update metric
                    for i, agent in enumerate(env.agents):
                        if env.helper_grid.get(*agent.pos).state == 1:
                            if in_selfish_region:
                                if env.in_selfish_region(*agent.pos, agent.index):
                                    mmdp_metric_ep[i] += 1
                            else:
                                mmdp_metric_ep[i] += 1

                    # check if episode is done
                    if done:
                        # normalize metric
                        mmdp_metric_ep /= t + 1
                        break

                    # process next observation
                    next_state = env.get_state()
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(
                        device
                    )
                    ma_obs = process_observation(next_obs, device, next_state)
                # reset environment to initial state
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)
                # update mmdp_metric_current_fire with metric for current episode
                mmdp_metric_current_fire += mmdp_metric_ep

            # average out metric values over all episodes
            mg_metric_current_fire /= self.episodes_per_state
            mmdp_metric_current_fire /= self.episodes_per_state

            # update final metric values with metric values for current initial fire location
            final_mg_metric += mg_metric_current_fire
            final_mmdp_metric += mmdp_metric_current_fire

        # average out final metric values over all initial states
        final_mg_metric /= len(self.initial_state_identifiers)
        final_mmdp_metric /= len(self.initial_state_identifiers)

        # save experiment data
        exp_data = self.exp_data.copy()
        exp_data["final_mg_metric"] = final_mg_metric.tolist()
        exp_data["final_mmdp_metric"] = final_mmdp_metric.tolist()
        with open(
            f"{mmdp_results_path}/{self.run}_exp_config.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(exp_data, fp, sort_keys=True, indent=4)
        with open(
            f"{mg_results_path}/{self.run}_exp_config.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(exp_data, fp, sort_keys=True, indent=4)

        return exp_data

    def make_spider_chart(self):
        """Create a radar chart to visualize the performance of agents."""

        # directories to store results
        mmdp_results_path = f"policy_eval/results/{self.mmdp_policy}_policy/strategy_metrics/spider_charts"
        mg_results_path = f"policy_eval/results/{self.mg_policy}_policy/strategy_metrics/spider_charts"
        if not os.path.exists(mmdp_results_path):
            os.makedirs(mmdp_results_path)
        if not os.path.exists(mg_results_path):
            os.makedirs(mg_results_path)

        # compute metrics
        data_ba = self.boundary_attack_metric()
        data_do = self.distance_from_other_agents_metric()
        data_df = self.distance_from_selfish_region_metric()
        data_tof = self.time_over_fire_metric()
        data_tof_sr = self.time_over_fire_metric(in_selfish_region=True)

        # loop over agents to create a combined radar chart
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "polar"}] * 2] * 1, horizontal_spacing=0.2
        )
        theta = [
            r"$BA$",
            r"$d_{\text{agents}}$",
            r"$d_{\text{SR}}$",
            r"$TOF$",
            r"$TOF_{\text{SR}}$",
            r"$BA$",
        ]
        for i in range(self.env.num_agents):
            r_mmdp = [
                data_ba["final_mmdp_metric"][i],
                data_do["final_mmdp_metric"][i],
                data_df["final_mmdp_metric"][i],
                data_tof["final_mmdp_metric"][i],
                data_tof_sr["final_mmdp_metric"][i],
                data_ba["final_mmdp_metric"][i],
            ]
            r_mg = [
                data_ba["final_mg_metric"][i],
                data_do["final_mg_metric"][i],
                data_df["final_mg_metric"][i],
                data_tof["final_mg_metric"][i],
                data_tof_sr["final_mg_metric"][i],
                data_ba["final_mg_metric"][i],
            ]
            fig.add_trace(
                go.Scatterpolar(
                    theta=theta,
                    r=r_mg,
                    name=f"MG Agent {i}",
                    fill="toself",
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Scatterpolar(
                    theta=theta,
                    r=r_mmdp,
                    name=f"MMDP Agent {i}",
                    fill="toself",
                ),
                row=1,
                col=i + 1,
            )
        fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text}</b>"))
        fig.update_annotations(font=dict(family="Helvetica", size=12))
        fig.write_image(f"{mg_results_path}/{self.run}_spider_chart.png", scale=8)
        fig.write_image(f"{mmdp_results_path}/{self.run}_spider_chart.png", scale=8)
