import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box

# from marllib import marl
from wildfire_environment.envs import WildfireEnv

# from marllib.envs.base_env import ENV_REGISTRY
import time

# register all scenario with env class
REGISTRY = {}
REGISTRY["wildfire"] = WildfireEnv
grouping_policies = False

if grouping_policies:
    # provide detailed information of each scenario
    # mostly for policy sharing
    policy_mapping_dict = {
        "wildfire": {
            "description": "UAVs extinguish wildfire",
            "team_prefix": ("TeamA_", "TeamB_"),
            "all_agents_one_policy": True,  # Is all-sharing of parameters applicable to wildfire?
            "one_agent_one_policy": True,  # Is no-sharing of parameters applicable to wildfire?
        },
    }

    # must inherited from MultiAgentEnv class
    class RLlibWildfire(MultiAgentEnv):
        def __init__(self, env_config):
            map = env_config["map_name"]
            env_config.pop("map_name", None)

            self.env = REGISTRY[map](**env_config)
            # assume all agent same action/obs space
            self.action_space = self.env.action_space["0"]
            # assume fully observable env. Modify WildfireEnv and below for partial obs.
            # Modify below if global state and action mask keys are needed in obs.
            state_low = np.full(
                (self.env.obs_depth + 1) * (self.env.grid_size**2) + 1, 0
            )
            state_high = np.full(
                (self.env.obs_depth + 1) * (self.env.grid_size**2) + 1, 1
            )
            self.observation_space = GymDict(
                {
                    "obs": Box(
                        low=self.env.observation_space["0"].low,
                        high=self.env.observation_space["0"].high,
                        dtype=self.env.observation_space["0"].dtype,
                    ),
                    "state": Box(
                        low=state_low,
                        high=state_high,
                        dtype=self.env.observation_space["0"].dtype,
                    ),
                }
            )
            self.agents = [
                f"{a.index}" for a in self.env.agents
            ]  # assumes agents are named by index. WildfireEnv assumes self.agents is list of Agent objects, MARLlib assumes it is list of agent names, here name = agent ID.
            self.num_agents = self.env.num_agents
            agent_ls = []
            # simple case: separate agent into two groups
            for i in range(self.env.num_agents):
                if i < 2:
                    agent_ls.append("TeamA_{}".format(i))
                else:
                    agent_ls.append("TeamB_{}".format(i))
            self.agents_group = agent_ls
            env_config["map_name"] = map
            self.env_config = env_config

        def reset(self, state=None, seed=None):
            if state is not None:
                original_obs, _ = self.env.reset(state=state, seed=seed)
            else:
                original_obs, _ = self.env.reset(seed=seed)
            state = self.env.get_state()
            obs = {}
            # swap name
            for agent_origin_name, agent_name in zip(self.agents, self.agents_group):
                obs[agent_name] = {
                    "obs": np.array(original_obs[agent_origin_name]),
                    "state": state,
                }
            return obs

        def step(self, action_dict):
            actions = {
                self.agents[i]: action_dict[key]
                for i, key in enumerate(action_dict.keys())
            }
            o, r, d, i = self.env.step(actions)
            state = self.env.get_state()
            rewards = {}
            obs = {}
            infos = {}
            for agent_origin_name, agent_name in zip(self.agents, self.agents_group):
                rewards[agent_name] = r[agent_origin_name]
                obs[agent_name] = {
                    "obs": np.array(o[agent_origin_name]),
                    "state": state,
                }
                infos[agent_name] = i[agent_origin_name]
            dones = {
                "__all__": d
            }  # d and t is a single value for all agents in WildfireEnv. Modify if separate d, t for separate agents.
            return obs, rewards, dones, infos

        def close(self):
            self.env.close()

        def render(self, mode=None):
            frame = self.env.render()
            time.sleep(0.05)
            return frame

        def get_env_info(self):
            env_info = {
                "space_obs": self.observation_space,
                "space_act": self.action_space,
                "num_agents": self.num_agents,
                "episode_limit": self.env.max_steps,
                "policy_mapping_info": policy_mapping_dict,
            }
            return env_info

else:
    # provide detailed information of each scenario
    # mostly for policy sharing
    policy_mapping_dict = {
        "wildfire": {
            "description": "UAVs extinguish wildfire",
            "team_prefix": ("UAVs_"),
            "all_agents_one_policy": True,  # Is all-sharing of parameters applicable to wildfire?
            "one_agent_one_policy": True,  # Is no-sharing of parameters applicable to wildfire?
        },
    }

    # must inherited from MultiAgentEnv class
    class RLlibWildfire(MultiAgentEnv):
        def __init__(self, env_config):
            map = env_config["map_name"]
            env_config.pop("map_name", None)

            self.env = REGISTRY[map](**env_config)
            # assume all agent same action/obs space
            self.action_space = self.env.action_space["0"]
            # assume fully observable env. Modify WildfireEnv and below for partial obs.
            # Modify below if global state and action mask keys are needed in obs.
            # one at the end for time awareness
            state_low = np.full(
                (self.env.obs_depth + 1) * (self.env.grid_size**2) + 1, 0
            )
            state_high = np.full(
                (self.env.obs_depth + 1) * (self.env.grid_size**2) + 1, 1
            )
            self.observation_space = GymDict(
                {
                    "obs": Box(
                        low=self.env.observation_space["0"].low,
                        high=self.env.observation_space["0"].high,
                        dtype=self.env.observation_space["0"].dtype,
                    ),
                    "state": Box(
                        low=state_low,
                        high=state_high,
                        dtype=self.env.observation_space["0"].dtype,
                    ),
                }
            )
            self.agents = [
                f"{a.index}" for a in self.env.agents
            ]  # assumes agents are named by index. WildfireEnv assumes self.agents is list of Agent objects, MARLlib assumes it is list of agent names, here name = agent ID.
            self.num_agents = self.env.num_agents
            env_config["map_name"] = map
            self.env_config = env_config

        def reset(self, state=None):
            if state is not None:
                original_obs, _ = self.env.reset(state=state)
            else:
                original_obs, _ = self.env.reset()
            state = self.env.get_state()
            obs = {}

            for name in self.agents:
                obs[name] = {
                    "obs": np.array(original_obs[name]),
                    "state": state,
                }
            return obs

        def step(self, action_dict):
            actions = {
                self.agents[i]: action_dict[key]
                for i, key in enumerate(action_dict.keys())
            }
            o, r, d, infos = self.env.step(actions)
            state = self.env.get_state()
            rewards = {}
            obs = {}
            for i, key in enumerate(action_dict.keys()):
                rewards[key] = r[self.agents[i]]
                obs[key] = {
                    "obs": np.array(o[self.agents[i]]),
                    "state": state,
                }
            dones = {
                "__all__": d
            }  # d and t is a single value for all agents in WildfireEnv. Modify if separate d, t for separate agents.
            return obs, rewards, dones, infos

        def close(self):
            self.env.close()

        def render(self, mode=None):
            frame = self.env.render()
            time.sleep(0.05)
            return frame

        def get_env_info(self):
            env_info = {
                "space_obs": self.observation_space,
                "space_act": self.action_space,
                "num_agents": self.num_agents,
                "episode_limit": self.env.max_steps,
                "policy_mapping_info": policy_mapping_dict,
            }
            return env_info
