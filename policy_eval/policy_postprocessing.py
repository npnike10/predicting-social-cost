import gym
import wildfire_environment
from agent_metrics import AgentMetrics
from social_cost import compute_social_costs
from ccm import save_time_series, latent_ccm
from render import render


# instantiate environment
env = gym.make(
    "wildfire-v0",
    size=17,
    alpha=0.15,
    beta=0.9,
    delta_beta=0.7,
    num_agents=2,
    agent_start_positions=((4, 8), (12, 8)),
    initial_fire_size=3,
    max_steps=100,
    cooperative_reward=True,
    selfish_region_xmin=[2, 11],
    selfish_region_xmax=[6, 13],
    selfish_region_ymin=[7, 6],
    selfish_region_ymax=[9, 10],
    log_selfish_region_metrics=True,
    render_selfish_region_boundaries=True,
)

# parameters
gamma = 0.99  # discount factor
num_episodes = 500  # number of episodes to perform for policy evaluation or agent metrics computation
initial_state_identifiers = [
    (13, 8),
]  # specifies the initial states over which to average the state visitation frequencies
mmdp_policy = "ippo_13Aug_run7"
mg_policy = "ippo_13Aug_run8"
stochastic_policy = True
# directories needed to load agent policies
mg_model_path = "exp_results/wildfire/ippo_test_13Aug_run2/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_8ffbe_00000_0_2024-09-01_23-12-26/checkpoint_001386/checkpoint-1386"
mg_params_path = "exp_results/wildfire/ippo_test_13Aug_run2/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_8ffbe_00000_0_2024-09-01_23-12-26/params copy.json"
mmdp_model_path = "exp_results/wildfire/ippo_test_13Aug_run1/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_0d215_00000_0_2024-09-01_23-15-56/checkpoint_001408/checkpoint-1408"
mmdp_params_path = "exp_results/wildfire/ippo_test_13Aug_run1/ippo_mlp_wildfire/IPPOTrainer_wildfire_wildfire_0d215_00000_0_2024-09-01_23-15-56/params copy.json"
COMPUTE_AGENT_METRICS = True  # whether to compute agent metrics
COMPUTE_SOCIAL_COST = False  # whether to compute social cost
EVALUALTE_POLICY = False  # whether to evaluate policy
VALUE_ESTIMATION_VS_SAMPLES = False  # whether to run value estimation vs samples code
COMPUTE_CCM = False  # whether to compute CCM
CCM_TIME_SERIES = False  # whether to save agent position time series
RENDER = False  # whether to render the environment

if COMPUTE_AGENT_METRICS:
    initial_fire_vertices = [
        (14, 9),
        (15, 9),
        (15, 8),
        (14, 8),
    ]  # vertices to identify initial fire region in visitation heatmaps
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
    aviz.state_visitation_heatmaps(initial_fire_vertices)
    aviz.make_spider_chart()

if COMPUTE_SOCIAL_COST:
    reward_function_type = 'negative'
    compute_social_costs(
        env,
        reward_function_type,
        mmdp_policy,
        mg_policy,
        'first_run',
        'first_run',
        load_expected_values=True,
        mmdp_expected_value_run='first_run',
        mg_expected_value_run='first_run',
        mmdp_num_episodes_for_expected_value_computation=100000,
        mg_num_episodes_for_expected_value_computation=100000,
        tolerance=1e-1,
        annotate=False,
    )

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

if RENDER:
    env_mg = gym.make(
    "wildfire-v0",
    size=17,
    alpha=0.15,
    beta=0.9,
    delta_beta=0.7,
    num_agents=2,
    agent_start_positions=((12, 6), (12, 13)),
    initial_fire_size=3,
    max_steps=100,
    cooperative_reward=False,
    selfish_region_xmin=[11, 11],
    selfish_region_xmax=[13, 13],
    selfish_region_ymin=[5, 12],
    selfish_region_ymax=[7, 14],
    log_selfish_region_metrics=True,
    render_selfish_region_boundaries=True,
)
    render(env_mg, mg_policy, mg_model_path, mg_params_path, shared_policy=False, stochastic_policy=stochastic_policy, initial_state_identifier=initial_state_identifiers[0])
    render(env, mmdp_policy, mmdp_model_path, mmdp_params_path, shared_policy=True, stochastic_policy=stochastic_policy, initial_state_identifier=initial_state_identifiers[0])