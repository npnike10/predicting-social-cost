import numpy as np
from marllib import marl
from ray import tune

env = marl.make_env(environment_name="wildfire", map_name="wildfire")

# initialize algorithm and load hyperparameters
alg = marl.algos.ippo(hyperparam_source="test")

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(
    env,
    alg,
    {
        "core_arch": "mlp",
        "encode_layer": "128-128",
    },
)

# start learning + extra experiment settings if needed. remember to check ray.yaml before use
alg.fit(
    env,
    model,
    # restore_path={
    #     "params_path": "exp_results/wildfire/idqn_test_13Aug_run5/idqn_mlp_wildfire/IDQNTrainer_wildfire_wildfire_fff28_00000_0_2024-08-20_19-47-53/params.json",  # experiment configuration
    #     "model_path": "exp_results/wildfire/idqn_test_13Aug_run5/idqn_mlp_wildfire/IDQNTrainer_wildfire_wildfire_fff28_00000_0_2024-08-20_19-47-53/checkpoint_010000/checkpoint-10000",  # checkpoint path
    # },
    stop={"timesteps_total": 25000000},
    local_mode=False,
    num_gpus=3,
    num_workers=23,
    share_policy="individual",
    evaluation_interval=5,
    evaluation_num_episodes=20,
    checkpoint_freq=500,
    seed=np.random.randint(low=0, high=10000),
    local_dir="exp_results/wildfire/ippo_test_13Aug_run5",
)
