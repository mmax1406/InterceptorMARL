import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional
import os
from datetime import datetime
import torch

class MultiAgentTargetEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 n_agents: int = 3,
                 n_targets: int = 3,
                 world_size_x: float = 20.0,
                 world_size_y: float = 10.0,
                 max_velocity: float = 0.5,
                 agent_radius: float = 0.2,
                 target_radius: float = 0.3,
                 collision_penalty: float = 10.0,
                 distance_reward_scale: float = 1.0,
                 collision_distance_scale: float = 1.0,
                 max_steps: int = 25,
                 apply_disturbances: bool = False,
                 disturbance_strength: float = 0.1,
                 disturbance_frequency: float = 0.1):

        super().__init__()
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.world_size_x = world_size_x
        self.world_size_y = world_size_y
        self.max_velocity = max_velocity
        self.agent_radius = agent_radius
        self.target_radius = target_radius      
        self.max_steps = max_steps
        self.apply_disturbances = apply_disturbances
        self.disturbance_strength = disturbance_strength
        self.disturbance_frequency = disturbance_frequency

        self.collision_penalty = collision_penalty
        self.distance_reward_scale = distance_reward_scale
        self.collision_distance_scale = collision_distance_scale
        self.collision_threshold = collision_distance_scale * 2 * agent_radius
        self.collision_threshold_sq = self.collision_threshold ** 2  # Pre-compute squared threshold
        self.max_distance = np.sqrt(world_size_x**2 + world_size_y**2)
        self.inv_max_distance = 1.0 / self.max_distance  # Pre-compute reciprocal

        state_dim = 2 * n_agents + 2 * n_targets + 2 * n_agents + n_targets
        low = -max(world_size_x, world_size_y)
        high = max(world_size_x, world_size_y)
        self._obs_space = spaces.Box(low=low, high=high, shape=(state_dim,), dtype=np.float32)
        self._single_action_space = spaces.Box(low=-max_velocity, high=max_velocity, shape=(2,), dtype=np.float32)

        self.action_spaces = {a: self._single_action_space for a in self.agents}
        self.observation_spaces = {a: self._obs_space for a in self.agents}
        self._combined_action_space = spaces.Box(low=-max_velocity, high=max_velocity, shape=(self.n_agents * 2,), dtype=np.float32)

        # Pre-allocate ALL buffers with appropriate dtypes
        self.agent_positions = np.zeros((n_agents, 2), dtype=np.float32)
        self.agent_velocities = np.zeros((n_agents, 2), dtype=np.float32)
        self.target_positions = np.zeros((n_targets, 2), dtype=np.float32)
        self.targets_reached = np.zeros(n_targets, dtype=bool)
        self.steps = 0

        # Pre-allocated computation buffers
        self._dist_agents_sq = np.zeros((n_agents, n_agents), dtype=np.float32)  # Store squared distances
        self._dist_targets = np.zeros((n_agents, n_targets), dtype=np.float32)
        self._rewards = np.zeros(n_agents, dtype=np.float32)
        self._current_velocities = np.zeros((n_agents, 2), dtype=np.float32)
        
        # Pre-allocate observation buffers to avoid repeated allocations
        self._obs_dict = {agent: np.zeros(state_dim, dtype=np.float32) for agent in self.agents}
        
        # Pre-allocate for observation computation
        self._rel_agent_pos = np.zeros((n_agents, 2), dtype=np.float32)
        self._rel_target_pos = np.zeros((n_targets, 2), dtype=np.float32)
        
        # Pre-allocate difference arrays for distance calculations
        self._diff_agents = np.zeros((n_agents, n_agents, 2), dtype=np.float32)
        self._diff_targets = np.zeros((n_agents, n_targets, 2), dtype=np.float32)
        
        # Pre-compute position bounds for clipping
        self.pos_min = np.array([self.agent_radius, self.agent_radius], dtype=np.float32)
        self.pos_max = np.array([self.world_size_x - self.agent_radius, 
                                  self.world_size_y - self.agent_radius], dtype=np.float32)
        
        # Cache for agent indices (avoid repeated list operations)
        self._agent_indices = list(range(n_agents))
        
        # Pre-allocate collision mask
        self._collision_mask = np.zeros((n_agents, n_agents), dtype=bool)

    def observation_space(self, agent_id: str) -> spaces.Space:
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id: str) -> spaces.Space:
        return self.action_spaces[agent_id]

    def _initialize_positions(self):
        """Optimized position initialization with pre-allocated arrays"""
        # Initialize agents
        for i in range(self.n_agents):
            max_attempts = 100
            for _ in range(max_attempts):
                pos = np.random.uniform([self.agent_radius, self.agent_radius], 
                                      [0.1 + self.agent_radius, self.world_size_y - self.agent_radius]).astype(np.float32)
                if i == 0:
                    self.agent_positions[0] = pos
                    break
                # Vectorized distance check
                dist_sq = np.sum((self.agent_positions[:i] - pos) ** 2, axis=1)
                if np.all(dist_sq > (2 * self.agent_radius) ** 2):
                    self.agent_positions[i] = pos
                    break

        # Initialize targets
        for i in range(self.n_targets):
            max_attempts = 100
            for _ in range(max_attempts):
                pos = np.random.uniform([self.world_size_x/2 + self.target_radius, self.target_radius], 
                                      [self.world_size_x - self.target_radius, self.world_size_y - self.target_radius]).astype(np.float32)
                if i == 0:
                    self.target_positions[0] = pos
                    break
                # Vectorized distance check
                dist_sq = np.sum((self.target_positions[:i] - pos) ** 2, axis=1)
                if np.all(dist_sq > (2 * self.target_radius) ** 2):
                    self.target_positions[i] = pos
                    break

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self._initialize_positions()
        self.agent_velocities.fill(0)
        self.targets_reached.fill(False)
        self.steps = 0
        
        # Generate all observations at once
        for idx in self._agent_indices:
            self._make_agent_observation_inplace(idx)

        return self._obs_dict, self._get_info()

    def _make_agent_observation_inplace(self, agent_idx: int):
        """Optimized observation generation using pre-allocated buffers"""
        my_pos = self.agent_positions[agent_idx]

        # Compute relative positions in-place
        np.subtract(self.agent_positions, my_pos, out=self._rel_agent_pos)
        self._rel_agent_pos[agent_idx] = 0.0  # Zero out self position
        
        np.subtract(self.target_positions, my_pos, out=self._rel_target_pos)
        
        # Build observation directly into the pre-allocated buffer
        obs = self._obs_dict[self.agents[agent_idx]]
        pos_end = 2 * self.n_agents
        target_end = pos_end + 2 * self.n_targets
        vel_end = target_end + 2 * self.n_agents
        
        obs[:pos_end] = self._rel_agent_pos.ravel()
        obs[pos_end:target_end] = self._rel_target_pos.ravel()
        obs[target_end:vel_end] = self.agent_velocities.ravel()
        obs[vel_end:] = self.targets_reached

    def step(self, actions: Dict[str, np.ndarray]):
        """Optimized step function with minimal allocations"""
        # Fast action extraction and clipping
        for idx, agent in enumerate(self.agents):
            np.clip(actions[agent], -self.max_velocity, self.max_velocity, out=self.agent_velocities[idx])
        
        self.steps += 1
        
        # Apply velocities with optional disturbances
        if self.apply_disturbances and np.random.random() < self.disturbance_frequency:
            np.copyto(self._current_velocities, self.agent_velocities)
            self._current_velocities += np.random.uniform(-self.disturbance_strength, 
                                                          self.disturbance_strength, 
                                                          (self.n_agents, 2))
            self.agent_positions += self._current_velocities
        else:
            self.agent_positions += self.agent_velocities
        
        # Vectorized boundary clipping
        np.clip(self.agent_positions, self.pos_min, self.pos_max, out=self.agent_positions)
        
        # Calculate rewards
        self._calculate_reward_fast()       

        # --- Update targets reached ---
        # For each target, check if any agent is within target_radius
        for t in range(self.n_targets):
            if not self.targets_reached[t]:
                if np.any(self._dist_targets[:, t] < (self.target_radius*self.inv_max_distance)):
                    self.targets_reached[t] = True

        # Check termination conditions
        terminated_flag = self.targets_reached.all()
        truncated_flag = self.steps >= self.max_steps

        # Update observations
        for idx in self._agent_indices:
            self._make_agent_observation_inplace(idx)
     
        # Create return dictionaries (required by gym API)
        rewards = {agent: float(self._rewards[idx]) for idx, agent in enumerate(self.agents)}
        terminations = {agent: terminated_flag for agent in self.agents}
        truncations = {agent: truncated_flag for agent in self.agents}
        infos = {agent: self._get_info() for agent in self.agents}
        
        return self._obs_dict, rewards, terminations, truncations, infos

    def _calculate_reward_fast(self):
        """Highly optimized reward calculation using squared distances and vectorization"""
        self._rewards.fill(0)
        
        # Calculate squared distances between agents (avoid sqrt when possible)
        np.subtract.outer(self.agent_positions[:, 0], self.agent_positions[:, 0], 
                         out=self._diff_agents[:, :, 0])
        np.subtract.outer(self.agent_positions[:, 1], self.agent_positions[:, 1], 
                         out=self._diff_agents[:, :, 1])
        np.sum(self._diff_agents ** 2, axis=2, out=self._dist_agents_sq)
        
        # Calculate distances to targets (need actual distances for normalized reward)
        np.subtract(self.agent_positions[:, None, :], self.target_positions[None, :, :], 
                   out=self._diff_targets)
        np.sqrt(np.sum(self._diff_targets ** 2, axis=2), out=self._dist_targets)
        
        # Normalize and clip distances
        self._dist_targets *= self.inv_max_distance
        np.clip(self._dist_targets, 0.0, 1.0, out=self._dist_targets)

        # Distance-based rewards (vectorized sum)
        np.sum(self._dist_targets, axis=1, out=self._rewards)
        self._rewards *= -1
        
        # Collision penalties using squared distances (faster than sqrt)
        np.less(self._dist_agents_sq, self.collision_threshold_sq, out=self._collision_mask)
        np.fill_diagonal(self._collision_mask, False)
        collision_counts = np.sum(self._collision_mask, axis=1)
        self._rewards -= collision_counts * 10
        
        self._rewards -= 0.01

        # Completion bonus
        if self.targets_reached.all():
            self._rewards += self.n_agents * 3

    def _get_info(self):
        """Return info dictionary with minimal copying"""
        return {
            'targets_reached': self.targets_reached.copy(),
            'num_targets_reached': int(self.targets_reached.sum()),
            'steps': int(self.steps),
            'agent_positions': self.agent_positions.copy(),
            'target_positions': self.target_positions.copy(),
        }

    def render(self, mode: str = 'human'):
        if mode == 'human':
            print(f"Step: {self.steps}")
            print(f"Targets reached: {np.sum(self.targets_reached)}/{self.n_targets}")
            print(f"Agent positions:\n{self.agent_positions}")
            print(f"Target positions:\n{self.target_positions}")
            print("-" * 50)

    def save_animation(self, policy, filename=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import imageio

        script_dir = os.path.dirname(os.path.abspath(__file__))
        animations_dir = os.path.join(script_dir, 'Animations')
        os.makedirs(animations_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if filename is None:          
            filename = f'multi_agent_env_{timestamp}.gif'
        else:
            filename = f'{filename}_multi_agent_env.gif'

        save_path = os.path.join(animations_dir, filename)

        obs, _ = self.reset()
        frames = []
        
        mpl.rcParams.update({
            "axes.facecolor": "#f8f9fa",
            "figure.facecolor": "#f8f9fa",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.edgecolor": "#cccccc",
            "font.family": "DejaVu Sans",
            "font.size": 11,
        })

        for step in range(self.max_steps):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_xlim(0, self.world_size_x)
            ax.set_ylim(0, self.world_size_y)
            ax.set_aspect('equal')
            ax.set_title(f"Simulation Step {self.steps} | Mean reward:{round(np.average(self._rewards),3)}", 
                        fontsize=13, fontweight="bold", color="#333")

            unreached = self.target_positions[~self.targets_reached]
            reached = self.target_positions[self.targets_reached]

            if len(unreached) > 0:
                ax.scatter(unreached[:, 0], unreached[:, 1], s=50, zorder=3)
            if len(reached) > 0:
                ax.scatter(reached[:, 0], reached[:, 1], s=50, alpha=0.6, zorder=2)

            ax.scatter(self.agent_positions[:, 0], self.agent_positions[:, 1],
                       s=180, edgecolors='white', linewidths=1.5, zorder=4)

            # Vectorized arrow rendering check
            vel_norms = np.linalg.norm(self.agent_velocities, axis=1)
            for pos, vel, norm in zip(self.agent_positions, self.agent_velocities, vel_norms):
                if norm > 1e-6:
                    ax.arrow(pos[0], pos[1], vel[0] * 2, vel[1] * 2, 
                            head_width=0.25, head_length=0.2, alpha=0.8, zorder=5)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(frame)
            plt.close(fig)

            with torch.no_grad():

                cont_actions, raw_action = policy.getAction(obs)
                actions_dict = {agent_id: cont_actions[agent_id] for agent_id in self.agents}

                obs, rewards, terminations, truncations, infos = self.step(actions_dict)
                if np.average(self._rewards)>0:
                    print('No good')

            if all(terminations.values()) or all(truncations.values()):
                frames.extend([frames[-1]] * 10)
                break           

        imageio.mimsave(save_path, frames, duration=1)
        return save_path


if __name__ == "__main__":
    env = MultiAgentTargetEnv(
        n_agents=3,
        n_targets=3,
        max_velocity=0.5,
        apply_disturbances=False,
    )

    obs, info = env.reset()

    
    for i in range(100):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if all(terminated.values()) or all(truncated.values()):
            obs, info = env.reset()
    