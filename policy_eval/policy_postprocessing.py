import gym
import wildfire_environment
from agent_metrics import AgentMetrics


# instantiate environment
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
)
# parameters
gamma = 0.99  # discount factor
num_episodes = 1  # number of episodes to perform for policy evaluation or agent metrics computation
mmdp_policy = "ippo_13Aug_run5"
mg_policy = "ippo_13Aug_run6"
stochastic_policy = True
# directories needed to load agent policies
mg_model_path = "exp_results/wildfire/ippo_test_13Aug_run6/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_b9bd5_00000_0_2024-09-01_23-06-27/checkpoint_001384/checkpoint-1384"
mg_params_path = "exp_results/wildfire/ippo_test_13Aug_run6/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_b9bd5_00000_0_2024-09-01_23-06-27/params copy.json"
mmdp_model_path = "exp_results/wildfire/ippo_test_13Aug_run5/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_a28c2_00000_0_2024-09-01_23-05-48/checkpoint_001411/checkpoint-1411"
mmdp_params_path = "exp_results/wildfire/ippo_test_13Aug_run5/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_a28c2_00000_0_2024-09-01_23-05-48/params copy.json"
COMPUTE_AGENT_METRICS = True  # whether to compute agent metrics
COMPUTE_SOCIAL_COST = False  # whether to compute social cost
EVALUALTE_POLICY = False  # whether to evaluate policy
VALUE_ESTIMATION_VS_SAMPLES = False  # whether to run value estimation vs samples code

if COMPUTE_AGENT_METRICS:
    initial_state_identifiers = [
        (14, 2),
    ]  # specifies the initial states over which to average the state visitation frequencies
    initial_fire_vertices = [
        (14, 2),
        (15, 2),
        (15, 3),
        (14, 3),
    ]  # vertices to identify initial fire region in visitation heatmaps
    selfish_region_vertices = (
        [(7, 7), (10, 7), (10, 10), (7, 10)],
        [(13, 1), (16, 1), (16, 4), (13, 4)],
    )  # vertices to identify selfish region in visitation heatmaps
    aviz = AgentMetrics(
        env,
        gamma,
        num_episodes,
        initial_state_identifiers,
        mmdp_policy,
        mg_policy,
        stochastic_policy,
        mmdp_model_path,
        mmdp_params_path,
        mg_model_path,
        mg_params_path,
    )
    aviz.state_visitation_heatmaps(initial_fire_vertices, selfish_region_vertices)
    # aviz.make_spider_chart()
