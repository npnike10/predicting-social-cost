import sys
import os
import numpy as np
import gym
import wildfire_environment
from wildfire_environment.utils.misc import save_frames_as_gif
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_agent_policies,
    process_observation,
    select_action,
)  # pylint: disable=import-error

# initialize environment. Set the arguments as desired.
env = gym.make(
    "wildfire-v0",
    size=17,
    alpha=0.15,
    beta=0.9,
    delta_beta=0.7,
    num_agents=2,
    agent_start_positions=((8, 8), (14, 2)),
    initial_fire_size=3,
    max_steps=100,
    cooperative_reward=True,
    selfish_region_xmin=[7, 13],
    selfish_region_xmax=[9, 15],
    selfish_region_ymin=[7, 1],
    selfish_region_ymax=[9, 3],
    log_selfish_region_metrics=True,
    render_selfish_region_boundaries=True,
)

# directories needed to load agent policies
MODEL_PATH = "exp_results/wildfire/ippo_test_13Aug_run5/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_a28c2_00000_0_2024-09-01_23-05-48/checkpoint_001411/checkpoint-1411"
PARAMS_PATH = "exp_results/wildfire/ippo_test_13Aug_run5/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_a28c2_00000_0_2024-09-01_23-05-48/params copy.json"
SHARED_POLICY = True  # whether agents share the same policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOCHASTIC_POLICY = True  # whether policy is stochastic

# load policies
agent_policies = load_agent_policies(
    MODEL_PATH, PARAMS_PATH, shared_policy=SHARED_POLICY, num_agents=env.num_agents
)

# run episodes
obs, _ = env.reset()
state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
ma_obs = process_observation(obs, device, state)
frames = []
frames.append(env.render())
num_episodes = 1

for ep in range(num_episodes):
    while True:
        ma_action = select_action(
            ma_obs, agent_policies, env.num_agents, STOCHASTIC_POLICY
        )
        obs, reward, done, _ = env.step(ma_action)
        state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
        ma_obs = process_observation(obs, device, state)
        frames.append(env.render())
        if done:
            break
    # save GIF for current episodes
    save_frames_as_gif(frames, path="./", filename="wildfire", ep=ep, fps=2, dpi=320)
