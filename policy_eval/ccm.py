"""Code for Convergent Cross Mapping (CCM) analysis of agent position time series.
"""

import os
import sys
import json
import gym
import numpy as np
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (  # pylint: disable=import-error, wrong-import-position
    process_observation,
    generate_time_series
)
from wildfire_environment.utils.misc import (  # pylint: disable=wrong-import-position
    get_initial_fire_coordinates,
)

def save_time_series(
    num_episodes,
    env,
    mg_policy,
    mmdp_policy,
    mg_model_path,
    mg_params_path,
    mmdp_model_path,
    mmdp_params_path,
    handcrafted_policy=None,
    stochastic_policy=False,
    initial_state_identifier=None,
    demarcate_episodes=False,
    replicate_number=None,
    ):
    """Save agent position time series to .npy files.

    Parameters
    ----------
    num_episodes : int
        number of Monte Carlo episodes to run
    env : MultiGridEnv
        environment in which to run episodes
    model_path : str
        path to the file containing agent policies' models
    params_path : str
        path to the params file for the policy training run. The file contains a dictionary. The dictionary in file at params_path should not contain the key "callbacks" and corresponding value. A copy of the original params file may be used for this purpose.
    handcrafted_policy : str, optional
        list of actions specifying the handcrafted policy. If episode steps is greater than length of list, actions are chosen by looping over list. By default None.
    shared_policy : bool
        whether policy sharing among agents is enabled
    stochastic_policy : bool, optional
        whether policy is stochastic.
    initial_state_identifier : tuple[int,int], optional
        specifies initial state of the environment. If None, the initial state is sampled uniformly at random from the initial state distribution. By default None.
    demarcate_episodes : bool, optional
        whether to demarcate time series of different episodes using a string 'NA' in between each episode series. By default False.
    replicate_number : int, optional
        If True, this time series data is a replicate and file name is modified to indicated replicate number. By default None.
    """

    mg_a1_time_series, mg_a2_time_series = generate_time_series(num_episodes, env, mg_model_path, mg_params_path, False, handcrafted_policy=handcrafted_policy, stochastic_policy=stochastic_policy, initial_state_identifier=initial_state_identifier, demarcate_episodes=demarcate_episodes)
    mmdp_a1_time_series, mmdp_a2_time_series = generate_time_series(num_episodes, env, mmdp_model_path, mmdp_params_path, True, handcrafted_policy=handcrafted_policy, stochastic_policy=stochastic_policy, initial_state_identifier=initial_state_identifier, demarcate_episodes=demarcate_episodes)
    RESULTS_PATH = f"policy_eval/results/time_series/demarcated{demarcate_episodes}/{mg_policy}_&_{mmdp_policy}"
    os.makedirs(RESULTS_PATH, exist_ok=True)
    if replicate_number:
        np.save(f"{RESULTS_PATH}/mg_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy", mg_a1_time_series)
        np.save(f"{RESULTS_PATH}/mg_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy", mg_a2_time_series)
        np.save(f"{RESULTS_PATH}/mmdp_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy", mmdp_a1_time_series)
        np.save(f"{RESULTS_PATH}/mmdp_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy", mmdp_a2_time_series)
    else:
        np.save(f"{RESULTS_PATH}/mg_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy", mg_a1_time_series)
        np.save(f"{RESULTS_PATH}/mg_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy", mg_a2_time_series)
        np.save(f"{RESULTS_PATH}/mmdp_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy", mmdp_a1_time_series)
        np.save(f"{RESULTS_PATH}/mmdp_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy", mmdp_a2_time_series)
