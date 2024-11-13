"""Code for Convergent Cross Mapping (CCM) analysis of agent position time series.
"""

import os
import numpy as np
from utils import (  # pylint: disable=import-error, wrong-import-position
    generate_time_series
)
import matplotlib.pyplot as plt
from latentccm import causal_inf
import torch
from latentccm import DATADIR
from latentccm.datagen_utils import generate_Lorenz_data

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

class GRU_reconstruction(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.RNN = torch.nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
            self.output_layer = torch.nn.Linear(hidden_size, input_size)
        def forward(self,x):
            output, _  = self.RNN(x)
            output = self.output_layer(output)
            return output
        
        def hidden_only(self,x):
            output, _  = self.RNN(x)
            return output

def latent_ccm(num_epochs = 500, batch_size = 32):
    _, y,_ = generate_Lorenz_data()
    X = y[:,:3]
    Y = y[:,3:]

    X_tensor = torch.stack(torch.Tensor(X[::10,:]).chunk(1000))
    Y_tensor = torch.stack(torch.Tensor(Y[::10,:]).chunk(1000))

    datasetX = torch.utils.data.TensorDataset(X_tensor)
    datasetY = torch.utils.data.TensorDataset(Y_tensor)

    dlX = torch.utils.data.DataLoader(datasetX, batch_size = batch_size, shuffle = False)
    dlY = torch.utils.data.DataLoader(datasetY, batch_size = batch_size, shuffle = False)

    # Simplest train loop
    dls = {"X":dlX, "Y":dlY}
    hiddens = {}
    for side in ["X","Y"]:
        loss_criterion = torch.nn.MSELoss()
        model = GRU_reconstruction(hidden_size = 20)
        optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
        for epoch in range(num_epochs):
            train_loss = 0
            for i,b in enumerate(dls[side]):
                optimizer.zero_grad()
                y_hat = model(b[0])
                loss1 = loss_criterion(y_hat[:,:-1,:],b[0][:,1:,:])
                loss2 = loss_criterion(y_hat[:,:-2,:],b[0][:,2:,:])
                loss = (loss1)# + 0.5*loss2)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach()
            train_loss /= (i+1)
            if (epoch%10)==0:
                print(f"Training_loss at epoch {epoch}: {train_loss}")

        hidden_path = []
        for i,b in enumerate(dls[side]):
            optimizer.zero_grad()
            y_hat = model.hidden_only(b[0])#[:,:,:]
            hidden_path.append(y_hat.detach())

        hiddens[side] = torch.cat(hidden_path).reshape(-1,y_hat.shape[-1])

    sc1, sc2 = causal_inf.CCM_compute(hiddens["X"].numpy()[::10],hiddens["Y"].numpy()[::10])

    plt.figure()
    plt.plot(sc1,label="X->Y")
    plt.plot(sc2,label = "Y->X")
    plt.legend()
    plt.title("Cross Mapping from the Latent Process of Lorenz Dynamical Systems")
    plt.show()