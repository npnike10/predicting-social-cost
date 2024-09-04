import os
import sys
import json
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# parameters
GRID_SIZE = 17  # size of the square gridworld (gridworld includes walls)
INITIAL_FIRE_SIZE = 3  # side of the square shaped initial fire region
MMDP_POLICY = "23Aug_run2"
MG_POLICY = "23Aug_run2"
REWARD_FUNCTION_TYPE = "negative"  # whether the reward function is always positive or always negative. This is used to determine the acceptable range of the social costs.
MMDP_VALUE_FUNCTION_RUN = (
    "test_run"  # specifies the value function of MMDP policy to use
)
MG_VALUE_FUNCTION_RUN = "test_run"  # specifies the value function of MG policy to use
EXPECTED_VALUES = None  # list containing expected values for MMDP and MG policies, in that order. Expected value is the expectation of state value function over the initial state distribution.
RUN = "23Aug_run2_&_2"  # run name
RESULTS_PATH = "policy_eval/results/social_cost"  # directory to store results
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# load value functions
mmdp_value_path = f"policy_eval/results/{MMDP_POLICY}_policy/value_function_estimates/{MMDP_VALUE_FUNCTION_RUN}_value_function.json"
mg_value_path = f"policy_eval/results/{MG_POLICY}_policy/value_function_estimates/{MG_VALUE_FUNCTION_RUN}_value_function.json"
with open(mmdp_value_path, "r", encoding="utf-8") as fp:
    mmdp_value_function = json.load(fp)
with open(mg_value_path, "r", encoding="utf-8") as fp:
    mg_value_function = json.load(fp)

# store experiment configuration and results
exp_data = {
    "config": {
        "grid size": GRID_SIZE,
        "run": RUN,
        "MMDP policy": MMDP_POLICY,
        "MG policy": MG_POLICY,
        "reward function type": REWARD_FUNCTION_TYPE,
        "MMDP value function run": MMDP_VALUE_FUNCTION_RUN,
        "MG value function run": MG_VALUE_FUNCTION_RUN,
        "expected values": EXPECTED_VALUES,
    },
}

# initialize array to store social cost for each initial state. Social cost of initial state with identifier (i,j) is stored at index (j,i) in the array.
social_costs = np.zeros((GRID_SIZE - 2, GRID_SIZE - 2))
# if expected values are specified
if EXPECTED_VALUES:
    # compute state aggregated social cost, the expectation of social cost over initial state distribution.
    state_aggregated_social_cost = EXPECTED_VALUES[0] / EXPECTED_VALUES[1]
    # initialize array to store difference between social cost of each initial state and state aggregated social cost.
    delta_social_costs = np.zeros((GRID_SIZE - 2, GRID_SIZE - 2))
# loop over all initial states. (i,j), the initial state identifier is the position of the center cell of the fire square, if it is odd sized. If the fire square is even sized, the top-left corner cell is chosen as the initial state identifier.
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        # skip (i,j) which are not valid initial state identifiers. The criteria for validity is corresponding initial fire must be fully contained inside the grid.
        if INITIAL_FIRE_SIZE % 2 != 0:
            if (
                i < ((INITIAL_FIRE_SIZE - 1) / 2) + 1
                or j < ((INITIAL_FIRE_SIZE - 1) / 2) + 1
                or i >= (GRID_SIZE - 1) - ((INITIAL_FIRE_SIZE - 1) / 2)
                or j >= (GRID_SIZE - 1) - ((INITIAL_FIRE_SIZE - 1) / 2)
            ):
                continue
        else:
            if i >= ((GRID_SIZE - 1) - (INITIAL_FIRE_SIZE / 2)) or j >= (
                (GRID_SIZE - 1) - (INITIAL_FIRE_SIZE / 2)
            ):
                continue
        # store social cost for current initial state
        social_costs[j, i] = (
            mmdp_value_function[str((i, j))] / mg_value_function[str((i, j))]
        )
        if EXPECTED_VALUES:
            # store difference between social cost of current initial state and state aggregated social cost
            delta_social_costs[j, i] = abs(
                social_costs[j, i] - state_aggregated_social_cost
            )

# store experiment data
exp_data["social costs"] = social_costs.tolist()
if EXPECTED_VALUES:
    exp_data["state aggregated social cost"] = state_aggregated_social_cost

# check if social costs are valid and set range of color bar for heatmap
if REWARD_FUNCTION_TYPE == "positive":
    # check if social costs are valid. Social cost is valid if it is always greater than 1.
    min_social_cost = np.min(social_costs[np.nonzero(social_costs)])
    # raise exception if an invalid social cost is encountered
    if min_social_cost < 1:
        raise ValueError(
            "Encountered an invalid social cost. Social cost must be greater than or equal to 1 for positive valued reward function. Please check your value functions to ensure MMDP value is always greater than MG value at every initial state."
        )
    vmax = np.max(social_costs)
    vmin = 1
else:
    # check if social costs are valid. Social cost is valid if it is always less than 1.
    max_social_cost = np.max(social_costs)
    # raise exception if an invalid social cost is encountered
    if max_social_cost > 1:
        raise ValueError(
            "Encountered an invalid social cost. Social cost must be less than or equal to 1 for negative valued reward function. Please check your value functions to ensure MMDP value is always greater than MG value at every initial state."
        )
    vmax = 1
    vmin = 0

# set range of x and y ticks for heatmap
xticks = np.arange(
    1,
    GRID_SIZE - 1,
)
yticks = np.arange(
    1,
    GRID_SIZE - 1,
)

# create and save social cost heatmap, delta social cost heatmap and state aggregated social cost heatmap
plt.figure(dpi=400)
heatmap_plot = sns.heatmap(
    social_costs,
    xticklabels=xticks,
    yticklabels=yticks,
    cmap="Greens",
    annot=False,
    annot_kws={"fontsize": 6},
    vmin=vmin,
    vmax=vmax,
)
heatmap_plot.xaxis.tick_top()
heatmap_plot.set(
    xlabel="x-coordinate of initial state identifier",
    ylabel="y-coordinate of initial state identifier",
    title="Social Cost at Different Initial States",
)
plt.savefig(f"{RESULTS_PATH}/{RUN}_social_costs.png")

if EXPECTED_VALUES:
    plt.figure(dpi=400)
    heatmap_plot = sns.heatmap(
        delta_social_costs,
        xticklabels=xticks,
        yticklabels=yticks,
        cmap="Greens",
        annot=True,
        annot_kws={"fontsize": 6},
        vmin=vmin,
        vmax=vmax,
    )
    heatmap_plot.xaxis.tick_top()
    heatmap_plot.set(
        xlabel="x-coordinate of initial state identifier",
        ylabel="y-coordinate of initial state identifier",
        title="Abs. Difference b/w Social Cost and State Aggregated Social Cost",
    )
    plt.savefig(f"{RESULTS_PATH}/{RUN}_delta_social_costs.png")

    plt.figure(dpi=400)
    heatmap_plot = sns.heatmap(
        state_aggregated_social_cost * np.ones((GRID_SIZE - 2, GRID_SIZE - 2)),
        xticklabels=xticks,
        yticklabels=yticks,
        cmap="Greens",
        annot=True,
        annot_kws={"fontsize": 6},
        vmin=vmin,
        vmax=vmax,
    )
    heatmap_plot.xaxis.tick_top()
    heatmap_plot.set(
        xlabel="x-coordinate of initial state identifier",
        ylabel="y-coordinate of initial state identifier",
        title="State Aggregated Social Cost",
    )
    plt.savefig(f"{RESULTS_PATH}/{RUN}_state_agg_social_cost.png")

# save experiment data
with open(f"{RESULTS_PATH}/{RUN}_exp_data.json", "w", encoding="utf-8") as fp:
    json.dump(exp_data, fp, sort_keys=True, indent=4)
