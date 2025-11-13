import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv


class PursuitEvasionEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "intercept_v2"}

    def __init__(self, N_adversaries=2, M_good=2, width_ratio=5.0, render_mode=None):
        self.N_adversaries = N_adversaries
        self.M_good = M_good
        self.agents = [f"adversary_{i}" for i in range(N_adversaries)] + [
            f"good_{i}" for i in range(M_good)
        ]
        self.possible_agents = self.agents.copy()
        self.agent_radius = 0.04
        self.width_ratio = width_ratio
        self.render_mode = render_mode
        self.viewer = None

        # World dimensions
        self.size_x = width_ratio * 2
        self.size_y = 2

        # Pre-allocate arrays (hot path optimization)
        num_agents = len(self.agents)
        self.pos_array = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.vel_array = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.active_array = np.ones(self.num_agents, dtype=bool)

        # Pre-compute agent type masks (avoid string checks in hot loop)
        self.is_adversary = np.array(['adversary' in a for a in self.agents])
        self.is_good = ~self.is_adversary
        self.adv_indices = np.where(self.is_adversary)[0]
        self.good_indices = np.where(self.is_good)[0]

        # Agent index lookup
        self.agent_to_idx = {a: i for i, a in enumerate(self.agents)}

        # --- Action spaces ---
        self.action_spaces = {
            **{
                f"adversary_{i}": gym.spaces.Box(
                    low=np.array([-1.0, -1.0]),
                    high=np.array([1.0, 1.0]),
                    dtype=np.float32,
                )
                for i in range(N_adversaries)
            },
            **{
                f"good_{i}": gym.spaces.Box(
                    low=np.array([-1.0, -1.0]),
                    high=np.array([1.0, 1.0]),
                    dtype=np.float32,
                )
                for i in range(M_good)
            },
        }

        # --- Observation space ---
        obs_dim = 16
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.observation_spaces = {a: obs_space for a in self.agents}

    # -----------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()
        self.active_array[:] = True

        # Vectorized initialization
        self.pos_array[self.is_adversary, 0] = self.width_ratio - 0.2
        self.pos_array[self.is_adversary, 1] = np.random.uniform(-0.8, 0.8, self.N_adversaries)
        self.pos_array[self.is_good, 0] = -self.width_ratio + 0.2
        self.pos_array[self.is_good, 1] = np.random.uniform(-0.8, 0.8, self.M_good)

        self.vel_array[self.is_adversary] = [-0.02, 0.0]
        self.vel_array[self.is_good] = [0.02, 0.0]

        return self._get_obs(), {a: {} for a in self.agents}

    # -----------------------------------------------------
    def compute_rewards(self):
        """Optimized reward function with dense shaping"""

        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}

        # Get active agent positions
        active_good = self.good_indices[self.active_array[self.good_indices]]
        active_adv = self.adv_indices[self.active_array[self.adv_indices]]

        good_reward = 0.0
        adv_reward = 0.0

        # ========================================
        # 1. DENSE DISTANCE-BASED REWARDS
        # ========================================
        if len(active_good) > 0 and len(active_adv) > 0:
            # Compute all pairwise distances
            good_pos = self.pos_array[active_good][:, np.newaxis, :]  # (n_good, 1, 2)
            adv_pos = self.pos_array[active_adv][np.newaxis, :, :]  # (1, n_adv, 2)
            dists = np.linalg.norm(good_pos - adv_pos, axis=2)  # (n_good, n_adv)

            # Reward good agents for being close to adversaries
            min_dists_per_good = dists.min(axis=1)  # Each good's distance to nearest adv
            min_dists_per_adv = dists.min(axis=0)  # Each adv's distance to nearest good

            # Good agents: reward for closing distance (normalized)
            proximity_reward_good = -0.1 * min_dists_per_good.mean()  # Negative distance = good
            good_reward += proximity_reward_good

            # Adversaries: reward for maintaining distance
            proximity_reward_adv = 0.05 * min_dists_per_adv.mean()  # Positive distance = good
            adv_reward += proximity_reward_adv

            # ========================================
            # 2. COLLISION DETECTION (Terminal reward)
            # ========================================
            collisions = dists < 2 * self.agent_radius
            if collisions.any():
                good_hit_idx, adv_hit_idx = np.where(collisions)
                num_collisions = len(good_hit_idx)

                # Mark as inactive
                self.active_array[active_good[good_hit_idx]] = False
                self.active_array[active_adv[adv_hit_idx]] = False

                # Large terminal reward
                good_reward += 50.0 * num_collisions
                adv_reward -= 50.0 * num_collisions

        # ========================================
        # 3. ADVERSARY PROGRESS TOWARD GOAL
        # ========================================
        if len(active_adv) > 0:
            # Reward adversaries for moving left (toward goal)
            progress_reward = 0.02 * np.sum(self.vel_array[active_adv, 0] < 0)  # Count moving left
            adv_reward += progress_reward

            # Check boundary crossing
            boundary_reached = self.pos_array[active_adv, 0] < -self.width_ratio + 0.1
            num_reached = boundary_reached.sum()
            if num_reached > 0:
                self.active_array[active_adv[boundary_reached]] = False
                # Large terminal reward for reaching goal
                good_reward -= 100.0 * num_reached
                adv_reward += 100.0 * num_reached

        # ========================================
        # 4. COOPERATION BONUS (Optional)
        # ========================================
        if len(active_good) > 1 and len(active_adv) > 0:
            # Reward good agents for surrounding adversaries
            for adv_idx in active_adv:
                adv_pos = self.pos_array[adv_idx]
                good_positions = self.pos_array[active_good]

                # Check if good agents are on different sides (x-axis)
                x_diffs = good_positions[:, 0] - adv_pos[0]
                has_left = (x_diffs < 0).any()
                has_right = (x_diffs > 0).any()

                # Check if good agents are on different sides (y-axis)
                y_diffs = good_positions[:, 1] - adv_pos[1]
                has_above = (y_diffs > 0).any()
                has_below = (y_diffs < 0).any()

                # Reward for pincer formation
                if has_left and has_right:
                    good_reward += 0.05
                if has_above and has_below:
                    good_reward += 0.05

        # ========================================
        # 5. ENERGY EFFICIENCY (Discourage erratic movement)
        # ========================================
        if len(active_good) > 0:
            # Small penalty for high velocities (encourages smooth movement)
            vel_magnitudes = np.linalg.norm(self.vel_array[active_good], axis=1)
            good_reward -= 0.005 * vel_magnitudes.mean()

        if len(active_adv) > 0:
            vel_magnitudes = np.linalg.norm(self.vel_array[active_adv], axis=1)
            adv_reward -= 0.005 * vel_magnitudes.mean()

        # ========================================
        # 6. ASSIGN REWARDS TO AGENTS
        # ========================================
        for a in self.agents:
            if "good" in a:
                rewards[a] = good_reward
            else:
                rewards[a] = adv_reward

        # ========================================
        # 7. TERMINATION
        # ========================================
        if not self.active_array[self.is_adversary].any():
            for a in self.agents:
                terminations[a] = True
            self.agents = []

        return rewards, terminations

    # -----------------------------------------------------
    def step(self, actions):
        # Vectorized physics update
        action_array = np.zeros((self.num_agents, 2), dtype=np.float32)
        for a, act in actions.items():
            idx = self.agent_to_idx[a]
            if self.active_array[idx]:
                action_array[idx] = np.clip(act, -1.0, 1.0)

        # Update velocities and positions (vectorized)
        self.vel_array += 0.02 * action_array
        self.pos_array += self.vel_array

        # Clamp positions (vectorized)
        self.pos_array[:, 0] = np.clip(
            self.pos_array[:, 0],
            -self.width_ratio + self.agent_radius,
            self.width_ratio - self.agent_radius
        )
        self.pos_array[:, 1] = np.clip(
            self.pos_array[:, 1],
            -1 + self.agent_radius,
            1 - self.agent_radius
        )

        rewards, terminations = self.compute_rewards()

        return (
            self._get_obs(),
            rewards,
            terminations,
            {a: bool(self.active_array[self.agent_to_idx[a]]) for a in self.agents},
            {a: {} for a in self.agents},
        )

    # -----------------------------------------------------
    def _get_obs(self):
        obs = {}

        # Get active positions for bounding boxes (vectorized)
        active_adv_mask = self.is_adversary & self.active_array
        active_good_mask = self.is_good & self.active_array

        adv_positions = self.pos_array[active_adv_mask]
        good_positions = self.pos_array[active_good_mask]

        # Compute bounding boxes
        if len(adv_positions) > 0:
            adv_box = np.array([
                adv_positions[:, 0].min(),
                adv_positions[:, 0].max(),
                adv_positions[:, 1].min(),
                adv_positions[:, 1].max(),
            ], dtype=np.float32)
        else:
            adv_box = np.zeros(4, dtype=np.float32)

        if len(good_positions) > 0:
            good_box = np.array([
                good_positions[:, 0].min(),
                good_positions[:, 0].max(),
                good_positions[:, 1].min(),
                good_positions[:, 1].max(),
            ], dtype=np.float32)
        else:
            good_box = np.zeros(4, dtype=np.float32)

        # --- Build observation per agent ---
        for a in self.agents:
            idx = self.agent_to_idx[a]
            if not self.active_array[idx]:
                obs[a] = np.zeros(16, dtype=np.float32)
                continue

            own_pos = self.pos_array[idx]
            own_vel = self.vel_array[idx]

            # Find nearest agents (vectorized)
            if "good" in a:
                # For good agents: find nearest adversary and nearest other good
                other_adv_mask = active_adv_mask.copy()
                other_good_mask = active_good_mask.copy()
                other_good_mask[idx] = False
            else:
                # For adversaries: find nearest other adversary and nearest good
                other_adv_mask = active_adv_mask.copy()
                other_adv_mask[idx] = False
                other_good_mask = active_good_mask.copy()

            nearest_adv = np.zeros(2, dtype=np.float32)
            nearest_good = np.zeros(2, dtype=np.float32)

            if other_adv_mask.any():
                adv_pos = self.pos_array[other_adv_mask]
                dists = np.linalg.norm(adv_pos - own_pos, axis=1)
                nearest_adv = adv_pos[dists.argmin()] - own_pos

            if other_good_mask.any():
                good_pos = self.pos_array[other_good_mask]
                dists = np.linalg.norm(good_pos - own_pos, axis=1)
                nearest_good = good_pos[dists.argmin()] - own_pos

            obs[a] = np.concatenate([
                own_pos, own_vel, nearest_adv, nearest_good, adv_box, good_box
            ]).astype(np.float32)

        return obs

def env(**kwargs):
    return PursuitEvasionEnv(**kwargs)

if __name__ == '__main__':
    environment = env(N_adversaries=3, M_good=5, width_ratio=3.0)
    observations, info = environment.reset()

    for step in range(200):
        actions = {agent: environment.action_spaces[agent].sample() for agent in environment.agents}
        obs, rewards, term, trunc, infos = environment.step(actions)
        environment.render()
        if not environment.agents:
            print("Game over")
            break