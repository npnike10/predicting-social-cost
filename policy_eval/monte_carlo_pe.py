"""Learn the value function of a policy using naive Monte Carlo estimation.
"""

import os
import sys
import json
import multiprocessing as mp
import gym
import numpy as np
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (  # pylint: disable=import-error, wrong-import-position
    process_observation,
    run_episodes,
)
from wildfire_environment.utils.misc import (  # pylint: disable=wrong-import-position
    get_initial_fire_coordinates,
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
    agent_start_positions=((8, 8), (14, 2)),
    initial_fire_size=3,
    cooperative_reward=True,
)

# parameters
GAMMA = 0.99  # discount factor
NUM_EPISODES = (
    5000  # sample size to use for Monte Carlo estimate of value function at each state
)
NUM_WORKERS = 16  # number of workers to use for parallel processing
POLICY = "ippo_10Sep_run2"  # policy to evaluate
SHARED_POLICY = False  # whether agents share the same policy
STOCHASTIC_POLICY = True  # whether policy is stochastic
RUN = "first_run"  # run name
USE_LOGS = False  # whether to use saved return samples instead of collecting new samples to estimate value function
LOG_PATH = None  # path to logged return samples
results_path = f"policy_eval/results/{POLICY}_policy/value_function_estimates"  # directory to store results
if not os.path.exists(results_path):
    os.makedirs(results_path)
if USE_LOGS:
    LOG_PATH = ""
    with open(LOG_PATH, "r", encoding="utf-8") as fp:
        state_wise_returns = json.load(fp)

# directories needed to load agent policies
MODEL_PATH = "exp_results/wildfire/ippo_test_10Sep_run2/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_2d194_00000_0_2024-09-10_18-05-30/checkpoint_001738/checkpoint-1738"
PARAMS_PATH = "exp_results/wildfire/ippo_test_10Sep_run2/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_2d194_00000_0_2024-09-10_18-05-30/params copy.json"

# choose device on which PyTorch tensors will be allocated
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# store experiment configuration and results
exp_data = {
    "config": {
        "run": RUN,
        "num_episodes": NUM_EPISODES,
        "use_logs": USE_LOGS,
        "log_path": LOG_PATH,
        "input_model_dir": MODEL_PATH,
        "input_params_dir": PARAMS_PATH,
        "shared_policy": SHARED_POLICY,
        "stochastic_policy": STOCHASTIC_POLICY,
        "gamma": GAMMA,
    },
}

# initialize dictionary to store estimated value function for each state. The key is the state identifier and the value is the estimated value.
value_estimates = {}
if not USE_LOGS:
    # dictionary containing all return samples for each state. For re-use of sampled data to save redundant computational effort in future.
    state_wise_returns = {}

if __name__ == "__main__":
    grid_size = env.grid_size
    initial_fire_size = env.initial_fire_size
    mp.set_start_method("spawn")  # set the multiprocessing start method
    # loop over all initial states. (i,j), the initial state identifier is the position of the center cell of the fire square, if it is odd sized. If the fire square is even sized, the top-left corner cell is chosen as the initial state identifier.
    for i in tqdm(range(grid_size), desc="x-coordinate of initial state identifier"):
        for j in tqdm(
            range(grid_size),
            desc="y-coordinate of initial state identifier",
        ):
            # skip (i,j) which are not valid initial state identifiers. The criteria for validity is corresponding initial fire must be fully contained inside the grid.
            if env.initial_fire_size % 2 != 0:
                if (
                    i < ((initial_fire_size - 1) / 2) + 1
                    or j < ((initial_fire_size - 1) / 2) + 1
                    or i >= (grid_size - 1) - ((initial_fire_size - 1) / 2)
                    or j >= (grid_size - 1) - ((initial_fire_size - 1) / 2)
                ):
                    continue
            else:
                if i >= ((grid_size - 1) - (initial_fire_size / 2)) or j >= (
                    (grid_size - 1) - (initial_fire_size / 2)
                ):
                    continue
            if USE_LOGS:
                # current initial state
                initial_state_identifier = f"({i}, {j})"
                # get return samples for current initial state
                state_returns = state_wise_returns[initial_state_identifier]
                # calculate Monte Carlo estimate of value function for current initial state
                value_estimates[initial_state_identifier] = np.mean(
                    state_returns[: NUM_EPISODES + 1]
                )

            else:
                # get the positions of trees on fire in initial state
                trees_on_fire = get_initial_fire_coordinates(
                    i,
                    j,
                    grid_size,
                    initial_fire_size,
                )
                # reset env to initial state
                initial_state = torch.tensor(
                    env.construct_state(trees_on_fire, env.agent_start_positions, 0),
                    dtype=torch.float32,
                ).to(device)
                obs, _ = env.reset(state=initial_state.cpu().numpy())
                ma_obs = process_observation(obs, device, initial_state)

                # divide sample collection among workers
                samples_per_worker = [NUM_EPISODES // NUM_WORKERS] * NUM_WORKERS
                if NUM_EPISODES % NUM_WORKERS != 0:
                    samples_per_worker[-1] += NUM_EPISODES % NUM_WORKERS
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
                            initial_state,
                            device,
                            MODEL_PATH,
                            PARAMS_PATH,
                            SHARED_POLICY,
                            STOCHASTIC_POLICY,
                        )
                        for s in samples_per_worker
                    ],
                )

                # wait for all processes to finish
                pool.close()
                pool.join()

                # create single list containing all episode return samples
                all_return_samples = [
                    sample[0]
                    for worker_samples in collected_samples
                    for sample in worker_samples
                ]
                # initial state for all samples
                initial_state_identifier = f"{collected_samples[0][0][1]}"

                # calculate Monte Carlo estimate of value function for current initial state
                value_estimates[initial_state_identifier] = np.mean(all_return_samples)
                # store return samples for current initial state
                state_wise_returns[initial_state_identifier] = all_return_samples

    # save experiment data
    with open(f"{results_path}/{RUN}_value_function.json", "w", encoding="utf-8") as fp:
        json.dump(value_estimates, fp, sort_keys=True, indent=4)
    with open(f"{results_path}/{RUN}_exp_data.json", "w", encoding="utf-8") as fp:
        json.dump(exp_data, fp, sort_keys=True, indent=4)
    if not USE_LOGS:
        with open(
            f"{results_path}/{RUN}_state_wise_returns.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(state_wise_returns, fp, sort_keys=True, indent=4)
