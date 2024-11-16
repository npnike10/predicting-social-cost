import gym
import wildfire_environment

# from agent_metrics import AgentMetrics
from ccm import save_time_series, latent_ccm


# instantiate environment
env = gym.make(
    "wildfire-v0",
    size=17,
    alpha=0.15,
    beta=0.9,
    delta_beta=0.7,
    num_agents=2,
    agent_start_positions=((12, 6), (12, 13)),
    initial_fire_size=3,
    max_steps=100,
    cooperative_reward=True,
    selfish_region_xmin=[11, 11],
    selfish_region_xmax=[13, 13],
    selfish_region_ymin=[5, 12],
    selfish_region_ymax=[7, 14],
    log_selfish_region_metrics=True,
)
# parameters
gamma = 0.99  # discount factor
num_episodes = 1000  # number of episodes to perform for policy evaluation or agent metrics computation
initial_state_identifiers = [
    (7, 12),
]  # specifies the initial states over which to average the state visitation frequencies
mmdp_policy = "ippo_13Aug_run5"
mg_policy = "ippo_13Aug_run6"
stochastic_policy = True
# directories needed to load agent policies
mg_model_path = "exp_results/wildfire/ippo_test_13Aug_run12/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_7d8e9_00000_0_2024-09-02_17-27-08/checkpoint_001719/checkpoint-1719"
mg_params_path = "exp_results/wildfire/ippo_test_13Aug_run12/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_7d8e9_00000_0_2024-09-02_17-27-08/params copy.json"
mmdp_model_path = "exp_results/wildfire/ippo_test_13Aug_run11/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_f8c9c_00000_0_2024-09-02_17-23-25/checkpoint_001735/checkpoint-1735"
mmdp_params_path = "exp_results/wildfire/ippo_test_13Aug_run11/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_f8c9c_00000_0_2024-09-02_17-23-25/params copy.json"
COMPUTE_AGENT_METRICS = False  # whether to compute agent metrics
COMPUTE_SOCIAL_COST = False  # whether to compute social cost
EVALUALTE_POLICY = False  # whether to evaluate policy
VALUE_ESTIMATION_VS_SAMPLES = False  # whether to run value estimation vs samples code
COMPUTE_CCM = True  # whether to compute CCM
CCM_TIME_SERIES = False  # whether to save agent position time series

if COMPUTE_AGENT_METRICS:
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

if CCM_TIME_SERIES or COMPUTE_CCM:
    if CCM_TIME_SERIES:

        handcrafted_policy_scenario3 = [
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ]
        handcrafted_policy_scenario1 = [
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            4,
            4,
        ]
        handcrafted_policy_scenario5 = [
            4,
            4,
            4,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ]
        save_time_series(
            num_episodes,
            env,
            mg_policy,
            mmdp_policy,
            mg_model_path,
            mg_params_path,
            mmdp_model_path,
            mmdp_params_path,
            handcrafted_policy=handcrafted_policy_scenario5,
            stochastic_policy=stochastic_policy,
            initial_state_identifier=initial_state_identifiers[0],
            demarcate_episodes=False,
        )
    if COMPUTE_CCM:
        latent_ccm(
            mg_policy,
            mmdp_policy,
            num_episodes,
            initial_state_identifier=initial_state_identifiers[0],
            replicate_number=None,
            demarcated_episodes=False,
            episode_truncation_length=75,
        )
