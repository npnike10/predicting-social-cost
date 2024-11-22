import os
import sys
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# parameters
GRID_SIZE = 17  # size of the square gridworld (gridworld includes walls)
INITIAL_FIRE_SIZE = 3  # side of the square shaped initial fire region
MMDP_POLICY = "ippo_13Aug_run1"
MG_POLICY = "ippo_13Aug_run2"
REWARD_FUNCTION_TYPE = "negative"  # whether the reward function is always positive or always negative. This is used to determine the acceptable range of the social costs.
MMDP_VALUE_FUNCTION_RUN = (
    "first_run"  # specifies the value function of MMDP policy to use
)
MG_VALUE_FUNCTION_RUN = "first_run"  # specifies the value function of MG policy to use
LOAD_EXPECTED_VALUES = True  # whether to load the expected values (baseline method in paper) for MMDP and MG policies. Expected value is the expectation of state value function over the initial state distribution.
if LOAD_EXPECTED_VALUES:
    MMDP_EXPECTED_VALUE_RUN = "first_run"
    MG_EXPECTED_VALUE_RUN = "first_run"
    MMDP_NUM_EPISODES_FOR_EXPECTED_VALUE_COMPUTATION = 100000  # number of episodes used for Monte Carlo estimation of MMDP expected value
    MG_NUM_EPISODES_FOR_EXPECTED_VALUE_COMPUTATION = 100000  # number of episodes used for Monte Carlo estimation of MG expected value
RESULTS_PATH = f"policy_eval/results/social_cost_heatmaps/{MMDP_POLICY}_&_{MG_POLICY}"  # directory to store results
os.makedirs(RESULTS_PATH, exist_ok=True)
TOLERANCE = 1e-2  # tolerance for social cost validity check
ANNOTATE = False  # whether to annotate the heatmap with social cost values
if ANNOTATE:
    RUN += "_annotated"

# load value functions
mmdp_value_path = f"policy_eval/results/value_functions/{MMDP_POLICY}/{MMDP_VALUE_FUNCTION_RUN}_value_function.json"
mg_value_path = f"policy_eval/results/value_functions/{MG_POLICY}/{MG_VALUE_FUNCTION_RUN}_value_function.json"
with open(mmdp_value_path, "r", encoding="utf-8") as fp:
    mmdp_value_function = json.load(fp)
with open(mg_value_path, "r", encoding="utf-8") as fp:
    mg_value_function = json.load(fp)

# load expected value to serve as baseline to compare against social costs
if LOAD_EXPECTED_VALUES:
    expected_values_path = f"policy_eval/results/expected_value_function/{MMDP_POLICY}/{MMDP_EXPECTED_VALUE_RUN}_exp_data.json"
    with open(expected_values_path, "r", encoding="utf-8") as fp:
        mmdp_expected_value = json.load(fp)["value estimates"][str(MMDP_NUM_EPISODES_FOR_EXPECTED_VALUE_COMPUTATION)]
    expected_values_path = f"policy_eval/results/expected_value_function/{MG_POLICY}/{MG_EXPECTED_VALUE_RUN}_exp_data.json"
    with open(expected_values_path, "r", encoding="utf-8") as fp:
        mg_expected_value = json.load(fp)["value estimates"][str(MG_NUM_EPISODES_FOR_EXPECTED_VALUE_COMPUTATION)]

# store experiment configuration and results
exp_data = {
    "config": {
        "grid size": GRID_SIZE,
        "MMDP policy": MMDP_POLICY,
        "MG policy": MG_POLICY,
        "reward function type": REWARD_FUNCTION_TYPE,
        "MMDP value function run": MMDP_VALUE_FUNCTION_RUN,
        "MG value function run": MG_VALUE_FUNCTION_RUN,
        "load expected values": LOAD_EXPECTED_VALUES,
        "tolerance": TOLERANCE,
    },
}

# initialize array to store social cost for each initial state. Social cost of initial state with identifier (i,j) is stored at index (j,i) in the array.
social_costs = np.zeros((GRID_SIZE, GRID_SIZE))
# if expected values are specified
if LOAD_EXPECTED_VALUES:
    # compute state aggregated social cost, the expectation of social cost over initial state distribution.
    state_aggregated_social_cost = mmdp_expected_value / mg_expected_value
    # initialize array to store difference between social cost of each initial state and state aggregated social cost.
    delta_social_costs = np.zeros((GRID_SIZE, GRID_SIZE))
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
        if LOAD_EXPECTED_VALUES:
            # store difference between social cost of current initial state and state aggregated social cost
            delta_social_costs[j, i] = abs(
                social_costs[j, i] - state_aggregated_social_cost
            )


# check if social costs are valid and set range of color bar for heatmap
if REWARD_FUNCTION_TYPE == "positive":
    # check if social costs are valid. Social cost is valid if it is always greater than 1.
    min_social_cost = np.min(social_costs[np.nonzero(social_costs)])
    # raise exception if an invalid social cost is encountered
    if min_social_cost < (1 - TOLERANCE):
        print("Minimum value among social costs: ", min_social_cost)
        raise ValueError(
            "Encountered an invalid social cost. Social cost must be greater than or equal to 1 for positive valued reward function. Please check your value functions to ensure MMDP value is always greater than MG value at every initial state."
        )
    vmax = np.max(social_costs)
    vmin = 1
else:
    # check if social costs are valid. Social cost is valid if it is always less than 1.
    max_social_cost = np.max(social_costs)
    # raise exception if an invalid social cost is encountered
    if max_social_cost > (1 + TOLERANCE):
        print("Maximum value among social costs: ", max_social_cost)
        raise ValueError(
            "Encountered an invalid social cost. Social cost must be less than or equal to 1 for negative valued reward function. Please check your value functions to ensure MMDP value is always greater than MG value at every initial state."
        )
    vmax = 1
    vmin = 0

# set range of x and y ticks for heatmap
xticks = np.arange(
    0,
    GRID_SIZE,
)
yticks = np.arange(
    0,
    GRID_SIZE,
)

# Create a mask for boundary cells
mask = np.zeros_like(social_costs, dtype=bool)
mask[0, :] = True  # Top row
mask[-1, :] = True  # Bottom row
mask[:, 0] = True  # Left column
mask[:, -1] = True  # Right column

# Set boundary cells to NaN
social_costs_with_nan = social_costs.copy()
social_costs_with_nan[mask] = np.nan
if LOAD_EXPECTED_VALUES:
    delta_social_costs_with_nan = delta_social_costs.copy()
    delta_social_costs_with_nan[mask] = np.nan

# Create state aggregated social cost heatmap
if LOAD_EXPECTED_VALUES:
    state_aggregated_social_costs_with_nan = state_aggregated_social_cost * np.ones(
        (GRID_SIZE, GRID_SIZE)
    )
    state_aggregated_social_costs_with_nan[mask] = np.nan
    # zero out rows and columns where initial fire identifier cannot be located
    state_aggregated_social_costs_with_nan[1, :] = 0
    state_aggregated_social_costs_with_nan[:, 1] = 0
    state_aggregated_social_costs_with_nan[-2, :] = 0
    state_aggregated_social_costs_with_nan[:, -2] = 0

# Create a custom colormap
cmap = sns.color_palette("Greens", as_cmap=True)
cmap.set_bad(color="gray")  # Set NaN values to gray


# create and save social cost heatmap, delta social cost heatmap and state aggregated social cost heatmap
plt.figure(dpi=320)
heatmap_plot = sns.heatmap(
    social_costs_with_nan,
    xticklabels=xticks,
    yticklabels=yticks,
    cmap=cmap,
    annot=ANNOTATE,
    annot_kws={"fontsize": 5, "fontweight": "bold", "color": "white"},
    vmin=vmin,
    vmax=vmax,
)

heatmap_plot.xaxis.tick_top()
heatmap_plot.set(
    xlabel="x-coordinate of initial state identifier",
    ylabel="y-coordinate of initial state identifier",
    title="Social Cost at Different Initial States",
)


plt.savefig(f"{RESULTS_PATH}/social_costs.png")

if LOAD_EXPECTED_VALUES:
    plt.figure(dpi=320)
    heatmap_plot = sns.heatmap(
        delta_social_costs_with_nan,
        xticklabels=xticks,
        yticklabels=yticks,
        cmap=cmap,
        annot=ANNOTATE,
        annot_kws={"fontsize": 5, "fontweight": "bold", "color": "white"},
        vmin=vmin,
        vmax=vmax,
    )

    heatmap_plot.xaxis.tick_top()
    heatmap_plot.set(
        xlabel="x-coordinate of initial state identifier",
        ylabel="y-coordinate of initial state identifier",
        title="Abs. Difference b/w Social Cost and State Aggregated Social Cost",
    )
    plt.savefig(f"{RESULTS_PATH}/delta_social_costs.png")

    plt.figure(dpi=320)
    heatmap_plot = sns.heatmap(
        state_aggregated_social_costs_with_nan,
        xticklabels=xticks,
        yticklabels=yticks,
        cmap=cmap,
        annot=ANNOTATE,
        annot_kws={"fontsize": 5, "fontweight": "bold", "color": "white"},
        vmin=vmin,
        vmax=vmax,
    )

    heatmap_plot.xaxis.tick_top()
    heatmap_plot.set(
        xlabel="x-coordinate of initial state identifier",
        ylabel="y-coordinate of initial state identifier",
        title="State Aggregated Social Cost",
    )
    plt.savefig(f"{RESULTS_PATH}/state_agg_social_cost.png")

# store experiment data
exp_data["social costs"] = social_costs.tolist()
if LOAD_EXPECTED_VALUES:
    exp_data["state aggregated social cost"] = state_aggregated_social_cost
    # compute average delta social cost
    avg_delta_social_cost = np.nanmean(delta_social_costs)
    exp_data["averaged delta social cost"] = avg_delta_social_cost


# save experiment data
with open(f"{RESULTS_PATH}/exp_data.json", "w", encoding="utf-8") as fp:
    json.dump(exp_data, fp, sort_keys=True, indent=4)


