import os
import sys
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_social_costs(env, reward_function_type, mmdp_policy, mg_policy, mmdp_value_function_run, mg_value_function_run, load_expected_values=False, mmdp_expected_value_run=None, mg_expected_value_run=None, mmdp_num_episodes_for_expected_value_computation=None, mg_num_episodes_for_expected_value_computation=None, tolerance=1e-2, annotate=False):
    """ Compute social cost for each initial states and metrics for comparison against baseline state aggregated social cost.

    Parameters
    ----------
    env : WildfireEnv
        Wildfire environment
    reward_function_type : str  
        takes values 'positive' or 'negative'. specifies whether the reward function is always positive or always negative. This is used to determine the acceptable range of the social costs.
    mmdp_policy : str
        mmdp policy to compute social cost for
    mg_policy : str
        mg policy to compute social cost for
    mmdp_value_function_run : str
        specifies the value function of mmdp policy to use
    mg_value_function_run : str
        specifies the value function of mg policy to use
    load_expected_values : bool, optional
        whether to load the expected values (baseline method in paper) for mmdp and mg policies. Expected value is the expectation of state value function over the initial state distribution. By default False.
    mmdp_expected_value_run : str, optional
        run name for mmdp expected value computation, by default None.
    mg_expected_value_run : ste, optional
        run name for mg expected value computation, by default None
    mmdp_num_episodes_for_expected_value_computation : int, optional
        number of episodes used for Monte Carlo estimation of mmdp expected value, by default 100000
    mg_num_episodes_for_expected_value_computation : int, optional
        number of episodes used for Monte Carlo estimation of mg expected value, by default 100000
    tolerance : float, optional
        tolerance for social cost validity check, by default 1e-2
    annotate : bool, optional
        whether to annotate the heatmap with social cost values, by default False

    Raises
    ------
    ValueError
        if an invalid social cost value is encountered
    """
    # parameters
    GRID_SIZE = env.grid_size  # size of the square gridworld (gridworld includes walls)
    INITIAL_FIRE_SIZE = env.initial_fire_size  # side of the square shaped initial fire region
    RESULTS_PATH = f"policy_eval/results/social_cost_heatmaps/{mmdp_policy}_&_{mg_policy}"  # directory to store results
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # load value functions
    mmdp_value_path = f"policy_eval/results/value_functions/{mmdp_policy}/{mmdp_value_function_run}_value_function.json"
    mg_value_path = f"policy_eval/results/value_functions/{mg_policy}/{mg_value_function_run}_value_function.json"
    with open(mmdp_value_path, "r", encoding="utf-8") as fp:
        mmdp_value_function = json.load(fp)
    with open(mg_value_path, "r", encoding="utf-8") as fp:
        mg_value_function = json.load(fp)

    # load expected value to serve as baseline to compare against social costs
    if load_expected_values:
        expected_values_path = f"policy_eval/results/expected_value_function/{mmdp_policy}/{mmdp_expected_value_run}_exp_data.json"
        with open(expected_values_path, "r", encoding="utf-8") as fp:
            mmdp_expected_value = json.load(fp)["value estimates"][str(mmdp_num_episodes_for_expected_value_computation)]
        expected_values_path = f"policy_eval/results/expected_value_function/{mg_policy}/{mg_expected_value_run}_exp_data.json"
        with open(expected_values_path, "r", encoding="utf-8") as fp:
            mg_expected_value = json.load(fp)["value estimates"][str(mg_num_episodes_for_expected_value_computation)]

    # store experiment configuration and results
    exp_data = {
        "config": {
            "grid size": GRID_SIZE,
            "MMDP policy": mmdp_policy,
            "MG policy": mg_policy,
            "reward function type": reward_function_type,
            "MMDP value function run": mmdp_value_function_run,
            "MG value function run": mg_value_function_run,
            "load expected values": load_expected_values,
            "tolerance": tolerance,
        },
    }

    # initialize array to store social cost for each initial state. Social cost of initial state with identifier (i,j) is stored at index (j,i) in the array.
    social_costs = np.zeros((GRID_SIZE, GRID_SIZE))
    # if expected values are specified
    if load_expected_values:
        # compute state aggregated social cost, the expectation of social cost over initial state distribution.
        if reward_function_type == "positive":
            state_aggregated_social_cost = mmdp_expected_value / mg_expected_value
        else:
            state_aggregated_social_cost = mg_expected_value / mmdp_expected_value
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
            if reward_function_type == "positive":
                social_costs[j, i] = (
                    mmdp_value_function[str((i, j))] / mg_value_function[str((i, j))]
                )
            else:
                social_costs[j, i] = (
                    mg_value_function[str((i, j))] / mmdp_value_function[str((i, j))]
                )
            if load_expected_values:
                # store difference between social cost of current initial state and state aggregated social cost
                delta_social_costs[j, i] = abs(
                    social_costs[j, i] - state_aggregated_social_cost
                )


    # check if social costs are valid. Social cost is valid if it is always greater than 1.
    min_social_cost = np.min(social_costs[np.nonzero(social_costs)])
    # raise exception if an invalid social cost is encountered
    if min_social_cost < (1 - tolerance):
        print("Minimum value among social costs: ", min_social_cost)
        raise ValueError(
            "Encountered an invalid social cost. Social cost must be greater than or equal to 1 for positive valued reward function. Please check your value functions to ensure MMDP value is always greater than MG value at every initial state."
        )
    # set range of color bar for heatmap
    vmax = np.max(social_costs)
    vmin = 1

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
    if load_expected_values:
        delta_social_costs_with_nan = delta_social_costs.copy()
        delta_social_costs_with_nan[mask] = np.nan

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
        annot=annotate,
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

    # add selfish regions to heatmap
    for id, _ in enumerate(env.agents):
        anchor_point = (env.selfish_xmin[id], env.selfish_ymin[id])
        width = env.selfish_xmax[id] - env.selfish_xmin[id] + 1
        height =  env.selfish_ymax[id] - env.selfish_ymin[id] + 1
        selfish_region = patches.Rectangle(
                anchor_point,
                width,
                height,
                edgecolor=f"{env.agent_colors[id]}",
                facecolor="none",
                linewidth=0.5,
            )
        heatmap_plot.add_patch(selfish_region)

    if annotate:
        plt.savefig(f"{RESULTS_PATH}/social_costs_annotated.png")
    else:
        plt.savefig(f"{RESULTS_PATH}/social_costs.png")

    if load_expected_values:
        plt.figure(dpi=320)
        heatmap_plot = sns.heatmap(
            delta_social_costs_with_nan,
            xticklabels=xticks,
            yticklabels=yticks,
            cmap=cmap,
            annot=annotate,
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

        # add selfish regions to heatmap
        for id, _ in enumerate(env.agents):
            anchor_point = (env.selfish_xmin[id], env.selfish_ymin[id])
            width = env.selfish_xmax[id] - env.selfish_xmin[id] + 1
            height =  env.selfish_ymax[id]- env.selfish_ymin[id] + 1
            selfish_region = patches.Rectangle(
                    anchor_point,
                    width,
                    height,
                    edgecolor=f"{env.agent_colors[id]}",
                    facecolor="none",
                    linewidth=0.5,
                )
            heatmap_plot.add_patch(selfish_region)

        if annotate:
            plt.savefig(f"{RESULTS_PATH}/delta_social_costs_annotated.png")
        else:
            plt.savefig(f"{RESULTS_PATH}/delta_social_costs.png")

    # store experiment data
    exp_data["social costs"] = social_costs.tolist()
    if load_expected_values:
        exp_data["state aggregated social cost"] = state_aggregated_social_cost
        # compute average delta social cost
        avg_delta_social_cost = np.nanmean(delta_social_costs)
        max_delta_social_cost = np.nanmax(delta_social_costs)
        std_delta_social_cost = np.nanstd(delta_social_costs)
        exp_data["averaged delta social cost"] = avg_delta_social_cost
        exp_data["max delta social cost"] = max_delta_social_cost
        exp_data["std delta social cost"] = std_delta_social_cost


    # save experiment data
    with open(f"{RESULTS_PATH}/exp_data.json", "w", encoding="utf-8") as fp:
        json.dump(exp_data, fp, sort_keys=True, indent=4)


