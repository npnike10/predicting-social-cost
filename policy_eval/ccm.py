"""Code for Convergent Cross Mapping (CCM) analysis of agent position time series.
"""

import os
import numpy as np
from utils import (  # pylint: disable=import-error, wrong-import-position
    generate_time_series,
)
import matplotlib.pyplot as plt
from latentccm import causal_inf
import torch


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

    mg_a1_time_series, mg_a2_time_series = generate_time_series(
        num_episodes,
        env,
        mg_model_path,
        mg_params_path,
        False,
        handcrafted_policy=handcrafted_policy,
        stochastic_policy=stochastic_policy,
        initial_state_identifier=initial_state_identifier,
        demarcate_episodes=demarcate_episodes,
    )
    mmdp_a1_time_series, mmdp_a2_time_series = generate_time_series(
        num_episodes,
        env,
        mmdp_model_path,
        mmdp_params_path,
        True,
        handcrafted_policy=handcrafted_policy,
        stochastic_policy=stochastic_policy,
        initial_state_identifier=initial_state_identifier,
        demarcate_episodes=demarcate_episodes,
    )
    RESULTS_PATH = f"policy_eval/results/time_series/demarcated{demarcate_episodes}/{mg_policy}_&_{mmdp_policy}"
    os.makedirs(RESULTS_PATH, exist_ok=True)
    if replicate_number:
        np.save(
            f"{RESULTS_PATH}/mg_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy",
            mg_a1_time_series,
        )
        np.save(
            f"{RESULTS_PATH}/mg_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy",
            mg_a2_time_series,
        )
        np.save(
            f"{RESULTS_PATH}/mmdp_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy",
            mmdp_a1_time_series,
        )
        np.save(
            f"{RESULTS_PATH}/mmdp_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy",
            mmdp_a2_time_series,
        )
    else:
        np.save(
            f"{RESULTS_PATH}/mg_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy",
            mg_a1_time_series,
        )
        np.save(
            f"{RESULTS_PATH}/mg_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy",
            mg_a2_time_series,
        )
        np.save(
            f"{RESULTS_PATH}/mmdp_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy",
            mmdp_a1_time_series,
        )
        np.save(
            f"{RESULTS_PATH}/mmdp_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy",
            mmdp_a2_time_series,
        )


def load_time_series(
    datadir, num_episodes, initial_state_identifier=None, replicate_number=None
):
    """Load agent position time series from .npy files.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        agent position time series for the two policies in the two environments
    """
    if replicate_number:
        mg_a1_time_series = np.load(
            f"{datadir}/mg_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy",
            allow_pickle=True,
        )
        mg_a2_time_series = np.load(
            f"{datadir}/mg_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy",
            allow_pickle=True,
        )
        mmdp_a1_time_series = np.load(
            f"{datadir}/mmdp_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy",
            allow_pickle=True,
        )
        mmdp_a2_time_series = np.load(
            f"{datadir}/mmdp_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.npy",
            allow_pickle=True,
        )
    else:
        mg_a1_time_series = np.load(
            f"{datadir}/mg_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy",
            allow_pickle=True,
        )
        mg_a2_time_series = np.load(
            f"{datadir}/mg_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy",
            allow_pickle=True,
        )
        mmdp_a1_time_series = np.load(
            f"{datadir}/mmdp_a1_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy",
            allow_pickle=True,
        )
        mmdp_a2_time_series = np.load(
            f"{datadir}/mmdp_a2_time_series_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.npy",
            allow_pickle=True,
        )
    return (
        mg_a1_time_series,
        mg_a2_time_series,
        mmdp_a1_time_series,
        mmdp_a2_time_series,
    )


class GRU_reconstruction(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.RNN = torch.nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=1
        )
        self.output_layer = torch.nn.Linear(hidden_size, input_size)

    def forward(self, x):
        output, _ = self.RNN(x)
        output = self.output_layer(output)
        return output

    def hidden_only(self, x):
        output, _ = self.RNN(x)
        return output


def filter_time_series(a1_time_series, a2_time_series, episode_truncation_length):
    """Discard episodes that are too short.

    Parameters
    ----------
    a1_time_series : _type_
        _description_
    a2_time_series : _type_
        _description_
    episode_truncation_length : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    indices_to_delete = []
    for i, array in enumerate(a1_time_series):
        if len(array) < episode_truncation_length:
            indices_to_delete.append(i)
    a1_time_series = np.delete(a1_time_series, indices_to_delete, axis=0)
    a2_time_series = np.delete(a2_time_series, indices_to_delete, axis=0)

    a1_tensor = torch.stack(
        [
            torch.Tensor(array[:episode_truncation_length])
            for _, array in enumerate(a1_time_series)
        ]
    )
    a2_tensor = torch.stack(
        [
            torch.Tensor(array[:episode_truncation_length])
            for _, array in enumerate(a2_time_series)
        ]
    )
    return a1_tensor, a2_tensor


def save_plots(
    sc1,
    sc2,
    agents_type,
    num_episodes,
    resultsdir,
    initial_state_identifier=None,
    replicate_number=None,
):
    """Save plots of CCM results.

    Parameters
    ----------
    sc1 : np.ndarray
        CCM score for agent 1 position influence on agent 2 position
    sc2 : np.ndarray
        CCM score for agent 2 position influence on agent 1 position
    agents_type : str
        type of agents. For example "MG" or "MMDP"
    num_episodes : int
        number of episodes used to generate the time series
    initial_state_identifier : tuple[int,int]
        initial state of the environment used to generate the time series. If None, the initial state is sampled uniformly at random from the initial state distribution.
    replicate_number : int, optional
        If True, this time series data is a replicate. By default None.
    """
    plt.figure()
    plt.plot(sc1, label="Agent 1 Position -> Agent 2 Position")
    plt.plot(sc2, label="Agent 2 Position -> Agent 1 Position")
    plt.legend()
    plt.title(
        f"Cross Mapping from the Latent Process of Agent Positions in {agents_type}"
    )
    if replicate_number:
        plt.savefig(
            f"{resultsdir}/{agents_type}_lccm_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted_rep{replicate_number}.png"
        )
    else:
        plt.savefig(
            f"{resultsdir}/{agents_type}_lccm_{num_episodes}eps_{initial_state_identifier}fire_agent0Handcrafted.png"
        )


def compute_lccm(
    a1_time_series,
    a2_time_series,
    num_episodes,
    episode_truncation_length=0,
    num_epochs=500,
    gru_hidden_size=20,
    batch_size=32,
    lr=0.005,
    hiddens_skip_length=10,
):
    """Compute latent CCM."""
    # Truncate episodes to make them uniform in length
    if episode_truncation_length < 0:
        raise ValueError("episode_truncation_length must be a non-negative integer")
    if episode_truncation_length == 0:
        episode_truncation_length = min(
            [len(a1_time_series[i]) for i in range(num_episodes)]
        )

    # discard episodes that are too short, perform length truncation and convert to torch tensors
    if episode_truncation_length > 0:
        a1_tensor, a2_tensor = filter_time_series(
            a1_time_series, a2_time_series, episode_truncation_length
        )

    a1_dataset = torch.utils.data.TensorDataset(a1_tensor)
    a2_dataset = torch.utils.data.TensorDataset(a2_tensor)

    dl_a1 = torch.utils.data.DataLoader(
        a1_dataset, batch_size=batch_size, shuffle=False
    )
    dl_a2 = torch.utils.data.DataLoader(
        a2_dataset, batch_size=batch_size, shuffle=False
    )

    # Simplest train loop
    dls = {"Agent 1 Position": dl_a1, "Agent 2 Position": dl_a2}
    hiddens = {}
    for side in ["Agent 1 Position", "Agent 2 Position"]:
        loss_criterion = torch.nn.MSELoss()
        model = GRU_reconstruction(input_size=2, hidden_size=gru_hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            train_loss = 0
            for i, b in enumerate(dls[side]):
                optimizer.zero_grad()
                y_hat = model(b[0])
                loss1 = loss_criterion(y_hat[:, :-1, :], b[0][:, 1:, :])
                loss2 = loss_criterion(y_hat[:, :-2, :], b[0][:, 2:, :])
                loss = loss1  # + 0.5*loss2)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach()
            train_loss /= i + 1
            if (epoch % 10) == 0:
                print(f"Training_loss at epoch {epoch}: {train_loss}")

        hidden_path = []
        for i, b in enumerate(dls[side]):
            optimizer.zero_grad()
            y_hat = model.hidden_only(b[0])  # [:,:,:]
            hidden_path.append(y_hat.detach())

        hiddens[side] = torch.cat(hidden_path).reshape(-1, y_hat.shape[-1])

    print(hiddens["Agent 1 Position"].shape, hiddens["Agent 1 Position"][:5])
    sc1, sc2 = causal_inf.CCM_compute(
        hiddens["X"].numpy()[::hiddens_skip_length],
        hiddens["Y"].numpy()[::hiddens_skip_length],
    )
    return sc1, sc2


def latent_ccm(
    mg_policy,
    mmdp_policy,
    num_episodes,
    initial_state_identifier=None,
    replicate_number=None,
    demarcated_episodes=False,
    episode_truncation_length=0,
    num_epochs=500,
    gru_hidden_size=20,
    batch_size=32,
    lr=0.005,
):
    """Run Convergent Cross Mapping (CCM) analysis on agent position time series.

    Parameters
    ----------
    mg_policy : str
        MG policy name
    mmdp_policy : str
        MMDP policy name
    num_episodes : int
        number of Monte Carlo episodes to run
    initial_state_identifier : tuple[int,int], optional
        specifies initial state of the environment. If None, the initial state is sampled uniformly at random from the initial state distribution. By default None.
    replicate_number : int, optional
        If True, this time series data is a replicate and file name is modified to indicated replicate number. By default None.
    demarcated_episodes : bool, optional
        whether time series are demarcated using a string 'NA' in between each episode's data. By default False.
    episode_truncation_length : int, optional
        number of time steps to truncate each episode to. This is required to make the data from different episodes uniform in length to allow stacking into a dataset. By default 0, meaning length of shortest episode is used.
    num_epochs : int, optional
        number of epochs for training the GRU model. By default 500
    gru_hidden_size : int, optional
        hidden size for the GRU model. By default 20
    batch_size : int, optional
        batch size for training the GRU model. By default 32
    lr : float, optional
        learning rate for training the GRU model. By default 0.005
    """
    # Set up data and results directory
    datadir = f"policy_eval/results/time_series/demarcated{demarcated_episodes}/{mg_policy}_&_{mmdp_policy}"
    resultsdir = f"policy_eval/results/latent_ccm/{mg_policy}_&_{mmdp_policy}"
    os.makedirs(resultsdir, exist_ok=True)

    # Load time series data
    mg_a1_time_series, mg_a2_time_series, mmdp_a1_time_series, mmdp_a2_time_series = (
        load_time_series(
            datadir,
            num_episodes,
            initial_state_identifier=initial_state_identifier,
            replicate_number=replicate_number,
        )
    )

    # Compute latent CCM
    mg_sc1, mg_sc2 = compute_lccm(
        mg_a1_time_series,
        mg_a2_time_series,
        num_episodes,
        episode_truncation_length=episode_truncation_length,
        num_epochs=num_epochs,
        gru_hidden_size=gru_hidden_size,
        batch_size=batch_size,
        lr=lr,
    )
    mmdp_sc1, mmdp_sc2 = compute_lccm(
        mmdp_a1_time_series,
        mmdp_a2_time_series,
        num_episodes,
        episode_truncation_length=episode_truncation_length,
        num_epochs=num_epochs,
        gru_hidden_size=gru_hidden_size,
        batch_size=batch_size,
        lr=lr,
    )

    # Save plots
    save_plots(
        mg_sc1,
        mg_sc2,
        "MarkovGame",
        num_episodes,
        resultsdir,
        initial_state_identifier=initial_state_identifier,
        replicate_number=replicate_number,
    )
    save_plots(
        mmdp_sc1,
        mmdp_sc2,
        "MMDP",
        num_episodes,
        resultsdir,
        initial_state_identifier=initial_state_identifier,
        replicate_number=replicate_number,
    )
