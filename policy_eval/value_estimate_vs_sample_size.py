""" Get Monte Carlo estimate of value function for varying sample size. Initial state is sampled from the initial state distribution and same for all episodes.
"""

import os
import sys
import multiprocessing as mp
import ujson as json
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import wildfire_environment
from wildfire_environment.utils.misc import get_initial_fire_coordinates

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (  # pylint: disable=import-error, wrong-import-position
    process_observation,
    run_episodes,
)

# instantiate environment with cooperative reward.
env = gym.make(
    "wildfire-v0",
    max_steps=100,
    size=17,
    alpha=0.15,
    beta=0.9,
    delta_beta=0.7,
    num_agents=2,
    agent_start_positions=((12, 6), (12, 13)),
    initial_fire_size=3,
    cooperative_reward=True,
)

# parameters
sample_size_list = np.arange(
    1000, 250001, 1000
).tolist()  # list of sample sizes to use for Monte Carlo estimate of value function. One sample requires one episode.
GAMMA = 0.99  # discount factor
NUM_WORKERS = 16  # number of workers to use for parallel processing
POLICY = "ippo_23Aug_run2"  # policy to evaluate
SHARED_POLICY = False  # whether agents share the same policy
STOCHASTIC_POLICY = True  # whether policy is stochastic
INITIAL_STATE_IDENTIFIER = (
    2,
    2,
)  # initial state identifier for state to be evaluated. If None, initial state is sampled from initial state distribution.
ESTIMATE_EXPECTED_VALUE = False  # whether to estimate expected value. Expected value is the expectation of state value function over the initial state distribution.
run = "second_run"  # run name
if ESTIMATE_EXPECTED_VALUE:
    run = f"expected_value_{run}"
results_path = f"policy_eval/results/{POLICY}_policy/value_estimates_vs_num_samples"  # directory to store results
if not os.path.exists(results_path):
    os.makedirs(results_path)

# directories needed to load agent policies
MODEL_PATH = "exp_results/wildfire/ippo_mpe_23Aug_run2/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_f7c4a_00000_0_2024-08-30_17-34-07/checkpoint_001388/checkpoint-1388"
PARAMS_PATH = "exp_results/wildfire/ippo_mpe_23Aug_run2/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_f7c4a_00000_0_2024-08-30_17-34-07/params copy.json"

# choose device on which PyTorch tensors will be allocated
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# store experiment configuration and results
exp_data = {
    "config": {
        "run": run,
        "estimate_expected_value": ESTIMATE_EXPECTED_VALUE,
        "list of sample sizes": sample_size_list,
        "input_model_dir": MODEL_PATH,
        "input_params_dir": PARAMS_PATH,
        "shared_policy": SHARED_POLICY,
        "stochastic_policy": STOCHASTIC_POLICY,
        "gamma": GAMMA,
    },
}


# initialize dictionaries to store value estimates and standard deviations for different sample sizes.
value_estimates = {}
stdevs = {}

if __name__ == "__main__":
    if INITIAL_STATE_IDENTIFIER is not None:
        # construct initial state
        trees_on_fire = get_initial_fire_coordinates(
            INITIAL_STATE_IDENTIFIER[0],
            INITIAL_STATE_IDENTIFIER[1],
            env.grid_size,
            env.initial_fire_size,
        )
        state = torch.tensor(
            env.construct_state(trees_on_fire, env.agent_start_positions, 0),
            dtype=torch.float32,
        ).to(device)
        # reset env to specified initial state
        obs, _ = env.reset(state=state.cpu().numpy())
    else:
        # reset env to random initial state
        obs, _ = env.reset()
        state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
    # process observation for use with agent policies
    ma_obs = process_observation(obs, device, state)

    mp.set_start_method("spawn")  # set the multiprocessing start method
    # divide sample collection among workers
    samples_per_worker = [sample_size_list[-1] // NUM_WORKERS] * NUM_WORKERS
    if sample_size_list[-1] % NUM_WORKERS != 0:
        samples_per_worker[-1] += sample_size_list[-1] % NUM_WORKERS
    # collect samples
    pool = mp.Pool(processes=NUM_WORKERS)
    collected_samples = pool.starmap(
        run_episodes,
        [
            (
                s,
                env,
                GAMMA,
                ma_obs,
                state,
                device,
                MODEL_PATH,
                PARAMS_PATH,
                SHARED_POLICY,
                STOCHASTIC_POLICY,
                ESTIMATE_EXPECTED_VALUE,
            )
            for s in samples_per_worker
        ],
    )

    # wait for all processes to finish
    pool.close()
    pool.join()

    # create single list containing all episode return samples
    all_return_samples = [
        sample[0] for worker_samples in collected_samples for sample in worker_samples
    ]
    # create single list containing initial states for all samples
    all_initial_states = [
        f"{sample[1]}"
        for worker_samples in collected_samples
        for sample in worker_samples
    ]
    # calculate Monte Carlo estimates of value function for every sample size in sample_size_list
    for sample_size in sample_size_list:
        return_samples = all_return_samples[:sample_size]
        value_estimates[sample_size] = sum(return_samples) / sample_size
        stdevs[sample_size] = np.std(return_samples)

    # store experiment data
    exp_data["value estimates"] = value_estimates
    exp_data["standard deviations"] = stdevs

    # dictionary containing all return samples for each state. If estimate_expected_value is False, this dictionary will contain only one state.
    state_wise_returns = {}
    for i, state in enumerate(all_initial_states):
        if state not in state_wise_returns:
            state_wise_returns[state] = []
        state_wise_returns[state].append(all_return_samples[i])

    # save experiment data
    with open(f"{results_path}/{run}_exp_data.json", "w", encoding="utf-8") as fp:
        json.dump(exp_data, fp, sort_keys=True, indent=4)
    with open(
        f"{results_path}/{run}_state_wise_returns.json", "w", encoding="utf-8"
    ) as fp:
        json.dump(state_wise_returns, fp, sort_keys=True, indent=4)

    # create and save plot of value estimates against sample sizes
    plt.figure(0)
    plt.scatter(list(value_estimates.keys()), list(value_estimates.values()))
    plt.axhline(
        y=list(value_estimates.values())[-1] * 1.01,
        color="tab:gray",
        linestyle="dashed",
        linewidth=2,
        label=r"within 1% of estimate with largest sample size",
    )
    plt.axhline(
        y=list(value_estimates.values())[-1] * 0.99,
        color="tab:gray",
        linestyle="dashed",
        linewidth=2,
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("Value Estimate")
    plt.title("Monte Carlo Estimate of Value Function for Varying Sample Size")
    plt.legend()
    plt.savefig(f"{results_path}/{run}_plot.png")
