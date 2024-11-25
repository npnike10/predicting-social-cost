import sys
import os
import numpy as np
import gym
import wildfire_environment
from wildfire_environment.utils.misc import save_frames_as_gif, get_initial_fire_coordinates
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_agent_policies,
    process_observation,
    select_action,
)  # pylint: disable=import-error

def save_frames_as_gif(frames, path, filename, ep, fps=2, dpi=320):
    fig = plt.figure()
    plt.axis('off')
    ims = [[plt.imshow(frame, animated=True)] for frame in frames]
    anim = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
    gif_path = f"{path}{filename}-{ep}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def render(env, policy, model_path, params_path, shared_policy, stochastic_policy, initial_state_identifier=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # results directory
    RESULTS_PATH = f"policy_eval/results/videos/{policy}"  # directory to store results
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # load policies
    agent_policies = load_agent_policies(
        model_path, params_path, shared_policy=shared_policy, num_agents=env.num_agents
    )

    # run episodes
    if initial_state_identifier:
        # reset env to given initial state and get initial observations
        trees_on_fire = get_initial_fire_coordinates(
            *initial_state_identifier,
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
    frames = []
    frames.append(env.render())

    for ep in range(1):
        while True:
            ma_action = select_action(
                ma_obs, agent_policies, env.num_agents, stochastic_policy
            )
            obs, reward, done, _ = env.step(ma_action)
            state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
            ma_obs = process_observation(obs, device, state)
            frames.append(env.render())
            if done:
                break
        # save GIF for current episodes
        save_frames_as_gif(frames, path=RESULTS_PATH+"/", filename=f"{initial_state_identifier}fire-ep", ep=ep, fps=2, dpi=320)
