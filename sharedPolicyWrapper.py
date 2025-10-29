import numpy as np
import gymnasium as gym


class SharedPolicyWrapper(gym.Env):
    def __init__(self, pettingzoo_env):
        super().__init__()
        self.env = pettingzoo_env
        self.env.reset()
        self.agents = self.env.agents

        # Assume all agents share the same obs/action spaces
        self.observation_space = self.env.observation_space(self.agents[0])
        obs_sample = self.env.observation_space(self.agents[0]).sample()
        self.obs_dim = obs_sample.shape[0]
        self.action_space = self.env.action_space(self.agents[0])
        self.num_agents = len(self.agents)

        # Pre-allocate buffers to avoid repeated allocation
        self.obs_buf = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        self.reward_buf = np.zeros(self.num_agents, dtype=np.float32)
        self.action_buf = np.zeros((self.num_agents, self.action_space.shape[0]), dtype=np.float32)

        # Pre-identify adversary agents for faster lookup
        self.is_adversary = np.array(['adversary' in a for a in self.agents])

    def reset(self, *, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        # Reuse obs_buf instead of creating new array
        for i, a in enumerate(self.agents):
            self.obs_buf[i] = obs_dict[a]
        return self.obs_buf.copy(), {}

    def step(self, actions):
        # Vectorized action clipping - much faster than loop
        np.clip(actions, -1, 1, out=self.action_buf)

        # Apply adversary constraint only where needed
        if self.is_adversary.any():
            self.action_buf[self.is_adversary, 0] = np.clip(
                self.action_buf[self.is_adversary, 0], -1, 0
            )

        # Build action dict (unavoidable, but minimized)
        action_dict = {a: self.action_buf[i] for i, a in enumerate(self.agents)}

        next_obs, rewards, terms, truncs, infos = self.env.step(action_dict)

        # Check if done
        done = all(terms.values()) or len(self.env.agents) == 0

        # Handle finished case
        if len(next_obs) == 0 or done:
            dummy_obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
            # Fill reward buffer
            for i, a in enumerate(self.agents):
                self.reward_buf[i] = rewards.get(a, 0.0)
            infos["terminated"] = True
            return dummy_obs, self.reward_buf.copy(), True, infos

        # Reuse buffers instead of creating new arrays
        for i, a in enumerate(self.agents):
            if a in next_obs:  # Only update if agent still exists
                self.obs_buf[i] = next_obs[a]
            self.reward_buf[i] = rewards[a]

        return self.obs_buf.copy(), self.reward_buf.copy(), done, infos