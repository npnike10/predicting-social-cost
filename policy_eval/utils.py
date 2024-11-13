from itertools import count
from tqdm import tqdm
import torch
import numpy as np
from eval import load_model  # pylint: disable=import-error
from wildfire_environment.utils.misc import (  # pylint: disable=wrong-import-position
    get_initial_fire_coordinates,
)


def obs_to_torch(ma_obs, device):
    """Converts numpy arrays in ma_obs of RLlibWildfireEnv to torch tensors.

    Parameters
    ----------
    ma_obs : dict[str, dict[str, np.ndarray]]
        observation dictionary from RLlibWildfireEnv. Each key is the agent id and each value is a dictionary containing agent observation (the agent centered toroidal view of grid) and state.
    device : str
        the device on which PyTorch tensors will be allocated.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        observation dictionary with torch tensors.
    """
    for key in ma_obs:
        obs = ma_obs[key]
        for key2 in obs:
            obs[key2] = torch.from_numpy(np.array([obs[key2]])).to(device)
        ma_obs[key] = obs
    return ma_obs


def process_observation(ma_obs, device, state):
    """Constructs observation dictionary in the format of RLlibWildfireEnv from agent observations of WildfireEnv. In MARLlib observation dictionary, each key is the agent id and each value is a dictionary containing agent observation (the agent centered toroidal view of grid) and state.

    Parameters
    ----------
    ma_obs : dict[str, np.ndarray]
        observation dictionary from WildfireEnv. Each key is the agent id and each value is the agent observation (the agent centered toroidal view of grid).
    device : str
        the device on which PyTorch tensors will be allocated
    state : torch.Tensor
        the state of the environment

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        observation dictionary with torch tensors.
    """
    for key in ma_obs:
        ma_obs[key] = {"obs": torch.from_numpy(ma_obs[key]).to(device), "state": state}
    return ma_obs


def load_agent_policies(
    model_path: str, params_path: str, shared_policy: bool, num_agents: int
):
    """Load agent policies from training data.

    Parameters
    ----------
    model_path : str
        path to the file containing agent policies' models
    params_path : str
        path to the params file for the policy training run. The file contains a dictionary. The dictionary in file at params_path should not contain the key "callbacks" and corresponding value. A copy of the original params file may be used for this purpose.
    shared_policy : bool
        whether policy sharing among agents is enabled
    num_agents : int
        number of agents in the environment

    Returns
    -------
    list[TorchPolicy]
        list of agent policies, in order of increasing agent index.
    """
    # get checkpoint containing saved trainer
    ckpt = load_model(
        {
            "model_path": model_path,
            "params_path": params_path,
        }
    )
    # get policies from trainer
    if shared_policy:
        agent_policies = [
            ckpt.trainer.get_policy("shared_policy") for _ in range(num_agents)
        ]
    else:
        agent_policies = [
            ckpt.trainer.get_policy("policy_" + f"{i}") for i in range(num_agents)
        ]
    return agent_policies


def select_action(ma_obs, agent_policies, num_agents, stochastic_policy=False):
    """Get action for each agent for given observations and agent policies.

    Parameters
    ----------
    ma_obs :dict[str, dict[str, torch.Tensor]]
        observation dictionary from RLlibWildfireEnv. Each key is the agent id and each value is a dictionary containing agent observation (the agent centered toroidal view of grid) and state.
    agent_policies : list[TorchPolicy]
        list of agent policies, in order of increasing agent index.
    num_agents : int
        number of agents in the environment

    Returns
    -------
    dict[str, int]
        actions dictionary with agent id as key and agent action as value
    """
    if stochastic_policy:
        action = {
            f"{i}": agent_policies[i].compute_single_action(ma_obs[f"{i}"])[0]
            for i in range(num_agents)
        }
    else:
        action = {
            f"{i}": agent_policies[i].compute_single_action(
                ma_obs[f"{i}"], explore=False
            )[0]
            for i in range(num_agents)
        }
    return action


def run_episodes(
    num_episodes,
    env,
    gamma,
    ma_obs,
    state,
    device,
    model_path,
    params_path,
    shared_policy,
    stochastic_policy=False,
    estimate_expected_value=False,
):
    """Run specified number of Monte Carlo episodes in the environment starting from given state and following given agent policies.

    Parameters
    ----------
    num_episodes : int
        number of Monte Carlo episodes to run
    env : MultiGridEnv
        environment in which to run episodes
    gamma : int
        discount factor
    ma_obs : dict[str, dict[str, torch.Tensor]]
        initial observation dictionary where each key is the agent id and each value is a dictionary containing agent observation (the agent centered toroidal view of grid) and state.
    state : torch.Tensor
        the initial state of the environment for every episode
    device : str
        the device on which PyTorch tensors will be allocated
    model_path : str
        path to the file containing agent policies' models
    params_path : str
        path to the params file for the policy training run. The file contains a dictionary. The dictionary in file at params_path should not contain the key "callbacks" and corresponding value. A copy of the original params file may be used for this purpose.
    shared_policy : bool
        whether policy sharing among agents is enabled
    stochastic_policy : bool, optional
        whether policy is stochastic.
    estimate_expected_value : bool, optional
        whether to estimate expected value. If True, the initial state of the environment is sampled uniformly at random from the initial state distribution for each episode. By default False.

    Returns
    -------
    list[tuple[float,tuple[int,int]]]
        list containing return and initial state identifier for each episode. The initial state identifier is the position of the center cell of the fire square, if it is odd sized. If the fire square is even sized, the top-left corner cell is chosen as the initial state identifier. This identifier is only valid if initial state has a single square fire region. The purpose for saving this identifier is to be able to re-use the return samples to save redundant computational effort in future.
    """
    # load agent policies
    agent_policies = load_agent_policies(
        model_path, params_path, shared_policy=shared_policy, num_agents=env.num_agents
    )
    # initialize list to store returns
    episode_returns = []
    # run episodes
    for _ in tqdm(range(num_episodes), desc=f"Running {num_episodes} episodes"):
        # initialize return for current episode
        ret = 0
        # run one episode
        for t in count():
            # step the environment
            ma_action = select_action(
                ma_obs, agent_policies, env.num_agents, stochastic_policy
            )
            next_obs, reward, done, _ = env.step(ma_action)

            # update episode return
            ret += gamma**t * reward[f"{env.agents[0].index}"]

            # check if episode is done
            if done:
                break

            # process next observation
            next_state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
            ma_obs = process_observation(next_obs, device, next_state)
        # reset env for next episode
        if estimate_expected_value:
            # store return and initial state for current episode
            trees_on_fire = env.get_state_interpretation(
                state.cpu().numpy(), print_interpretation=False
            )[0]
            if env.initial_fire_size % 2 == 0:
                # if fire square is even sized, choose the top-left corner cell as the initial state identifier
                initial_state_identifier = trees_on_fire[0]
            else:
                initial_state_identifier = trees_on_fire[
                    (env.initial_fire_size**2 - 1) // 2
                ]
            episode_returns.append((ret, initial_state_identifier))
            # reset env to initial state sampled uniformly at random from initial state distribution
            obs, _ = env.reset()
            state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
            ma_obs = process_observation(obs, device, state)

        else:
            # store return and initial state for current episode
            trees_on_fire = env.get_state_interpretation(
                state.cpu().numpy(), print_interpretation=False
            )[0]
            if env.initial_fire_size % 2 == 0:
                # if fire square is even sized, choose the top-left corner cell as the initial state identifier
                initial_state_identifier = trees_on_fire[0]
            else:
                initial_state_identifier = trees_on_fire[
                    (env.initial_fire_size**2 - 1) // 2
                ]
            episode_returns.append((ret, initial_state_identifier))
            # reset env to specified initial state
            obs, _ = env.reset(state=state.cpu().numpy())
            ma_obs = process_observation(obs, device, state)
    return episode_returns

def generate_time_series(
    num_episodes,
    env,
    model_path,
    params_path,
    shared_policy,
    handcrafted_policy=None,
    stochastic_policy=False,
    initial_state_identifier=None,
    demarcate_episodes=False,
):
    """Run specified number of Monte Carlo episodes in the environment starting from given state and following given agent policies.

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
    shared_policy : bool
        whether policy sharing among agents is enabled
    handcrafted_policy : str, optional
        list of actions specifying the handcrafted policy. If episode steps is greater than length of list, actions are chosen by looping over list. By default None.
    stochastic_policy : bool, optional
        whether policy is stochastic.
    initial_state_identifier : tuple[int,int], optional
        specifies initial state of the environment. If None, the initial state is sampled uniformly at random from the initial state distribution. By default None.
    demarcate_episodes : bool, optional
        whether to demarcate time series of different episodes using a string 'NA' in between each episode series. By default False.
    
    Returns
    -------
    agent1_time_series : ndarray
        time series of agent 1 positions. ndarray of shape (N, 2) where N is the total number of time steps across all episodes and 2 is the number of coordinates in the position vector.
    agent2_time_series : ndarray
        time series of agent 2 positions. ndarray of shape (N, 2) where N is the total number of time steps across all episodes and 2 is the number of coordinates in the position vector.
    """

    # choose device on which PyTorch tensors will be allocated
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reset env
    if initial_state_identifier:
        trees_on_fire = get_initial_fire_coordinates(
            initial_state_identifier[0],
            initial_state_identifier[1],
            env.grid_size,
            env.initial_fire_size,
        )
        initial_state = torch.tensor(
            env.construct_state(trees_on_fire, env.agent_start_positions, 0),
            dtype=torch.float32,
        ).to(device)
        obs, _ = env.reset(state=initial_state.cpu().numpy())
        ma_obs = process_observation(obs, device, initial_state)
    else:
        obs, _ = env.reset()
        state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
        ma_obs = process_observation(obs, device, state)

    # load agent policies
    agent_policies = load_agent_policies(
        model_path, params_path, shared_policy=shared_policy, num_agents=env.num_agents
    )

    # initialize lists to store positions
    agent1_time_series = []
    agent2_time_series = []
    if not demarcate_episodes:
        agent1_time_series_array = np.empty((num_episodes,), dtype=object)
        agent2_time_series_array = np.empty((num_episodes,), dtype=object)

    # add initial positions to time series
    agent1_pos = list(env.agents[0].pos)
    agent2_pos = list(env.agents[1].pos)
    agent1_time_series.append(agent1_pos)
    agent2_time_series.append(agent2_pos)

    # run episodes
    for eps in tqdm(range(num_episodes), desc=f"Running {num_episodes} episodes"):
        
        # episode loop
        for t in count():
            # step the environment
            ma_action = select_action(
                ma_obs, agent_policies, env.num_agents, stochastic_policy
            )
            if handcrafted_policy:
                ma_action['0'] = handcrafted_policy[t % len(handcrafted_policy)]
            next_obs, reward, done, _ = env.step(ma_action)

            # update time series
            agent1_pos = list(env.agents[0].pos)
            agent2_pos = list(env.agents[1].pos)
            agent1_time_series.append(agent1_pos)
            agent2_time_series.append(agent2_pos)

            # check if episode is done
            if done:
                if initial_state_identifier:
                    obs, _ = env.reset(state=initial_state.cpu().numpy())
                    ma_obs = process_observation(obs, device, initial_state)
                else:
                    obs, _ = env.reset(state=state.cpu().numpy())
                    ma_obs = process_observation(obs, device, state)
                
                # add demarcation between episodes
                if demarcate_episodes:
                    agent1_time_series.append(["NA", "NA"])
                    agent2_time_series.append(["NA", "NA"])
                else:
                    agent1_time_series_array[eps] = np.array(agent1_time_series)
                    agent2_time_series_array[eps] = np.array(agent2_time_series)
                    agent1_time_series = []
                    agent2_time_series = []
                if eps<num_episodes-1:
                    # add initial positions to time series
                    agent1_pos = list(env.agents[0].pos)
                    agent2_pos = list(env.agents[1].pos)
                    agent1_time_series.append(agent1_pos)
                    agent2_time_series.append(agent2_pos)
                break

            # process next observation
            next_state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
            ma_obs = process_observation(next_obs, device, next_state)

    if demarcate_episodes:
        agent1_time_series = np.array(agent1_time_series)
        agent2_time_series = np.array(agent2_time_series)
    else:
        agent1_time_series = agent1_time_series_array
        agent2_time_series = agent2_time_series_array

    return agent1_time_series, agent2_time_series