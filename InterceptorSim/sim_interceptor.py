import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Optional


class AdversarialChaseEnv(gym.Env):
    """
    Adversarial multi-agent reinforcement learning environment.
    
    Two teams:
    - Pursuers (blue): Start on left, try to catch evaders
    - Evaders (red): Start on right, try to reach left side while avoiding pursuers
    
    Features:
    - Pursuers can move anywhere
    - Evaders can only move left or stay in place (x-velocity <= 0)
    - Continuous action space for both teams
    - Separate rewards for each team (zero-sum like)
    - Action repetition support
    - Episode ends when all evaders are caught or escaped
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        n_pursuers: int = 3,
        n_evaders: int = 3,
        world_size_x: float = 40.0,
        world_size_y: float = 20.0,
        max_velocity: float = 0.5,
        pursuer_radius: float = 0.2,
        evader_radius: float = 0.2,
        capture_distance: float = 0.5,
        escape_zone_x: float = 2.0,  # Left side safe zone width
        capture_reward: float = 100.0,
        escape_reward: float = 100.0,
        distance_reward_scale: float = 1.0,
        collision_penalty: float = -5.0,
        collision_distance_scale: float = 2.0,
        max_steps: int = 500,
        action_repeat: int = 5,
    ):
        super().__init__()
        
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.world_size_x = world_size_x
        self.world_size_y = world_size_y
        self.max_velocity = max_velocity
        self.pursuer_radius = pursuer_radius
        self.evader_radius = evader_radius
        self.capture_distance = capture_distance
        self.escape_zone_x = escape_zone_x
        self.capture_reward = capture_reward
        self.escape_reward = escape_reward
        self.distance_reward_scale = distance_reward_scale
        self.collision_penalty = collision_penalty
        self.collision_distance_scale = collision_distance_scale
        self.max_steps = max_steps
        self.action_repeat = action_repeat
        
        # State space: [pursuer_positions (2*N_p), evader_positions (2*N_e),
        #               pursuer_velocities (2*N_p), evader_velocities (2*N_e),
        #               evaders_caught (N_e), evaders_escaped (N_e)]
        state_dim = (2 * n_pursuers + 2 * n_evaders + 
                    2 * n_pursuers + 2 * n_evaders + 
                    n_evaders + n_evaders)
        self.observation_space = spaces.Box(
            low=-max(world_size_x, world_size_y),
            high=max(world_size_x, world_size_y),
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: velocity vectors for pursuers AND evaders
        # [pursuer_actions (2*N_p), evader_actions (2*N_e)]
        self.action_space = spaces.Box(
            low=-max_velocity,
            high=max_velocity,
            shape=((n_pursuers + n_evaders) * 2,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.pursuer_positions = np.zeros((n_pursuers, 2), dtype=np.float32)
        self.pursuer_velocities = np.zeros((n_pursuers, 2), dtype=np.float32)
        self.evader_positions = np.zeros((n_evaders, 2), dtype=np.float32)
        self.evader_velocities = np.zeros((n_evaders, 2), dtype=np.float32)
        self.evaders_caught = np.zeros(n_evaders, dtype=bool)
        self.evaders_escaped = np.zeros(n_evaders, dtype=bool)
        self.steps = 0
        
        # For reward calculation
        self.previous_pursuer_distances = np.zeros(n_pursuers, dtype=np.float32)
        self.previous_evader_distances = np.zeros(n_evaders, dtype=np.float32)
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset pursuers on left side
        self.pursuer_positions = np.random.uniform(
            low=[0, 0],
            high=[0.1, self.world_size_y],
            size=(self.n_pursuers, 2)
        ).astype(np.float32)
        
        # Reset evaders on right side
        self.evader_positions = np.random.uniform(
            low=[self.world_size_x * 0.8, 0],
            high=[self.world_size_x, self.world_size_y],
            size=(self.n_evaders, 2)
        ).astype(np.float32)
        
        # Reset velocities and status
        self.pursuer_velocities = np.zeros((self.n_pursuers, 2), dtype=np.float32)
        self.evader_velocities = np.zeros((self.n_evaders, 2), dtype=np.float32)
        self.evaders_caught = np.zeros(self.n_evaders, dtype=bool)
        self.evaders_escaped = np.zeros(self.n_evaders, dtype=bool)
        self.steps = 0
        
        # Calculate initial distances
        self._update_previous_distances()
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one action for both teams, repeated for action_repeat simulation steps.
        
        Args:
            action: Array containing [pursuer_actions, evader_actions]
        
        Returns:
            observation, (pursuer_reward, evader_reward), terminated, truncated, info
        """
        # Split actions for pursuers and evaders
        pursuer_action_size = self.n_pursuers * 2
        pursuer_actions = action[:pursuer_action_size].reshape(self.n_pursuers, 2)
        evader_actions = action[pursuer_action_size:].reshape(self.n_evaders, 2)
        
        # Clip actions to max velocity
        pursuer_actions = np.clip(pursuer_actions, -self.max_velocity, self.max_velocity)
        evader_actions = np.clip(evader_actions, -self.max_velocity, self.max_velocity)
        
        # IMPORTANT: Evaders can only move left (negative x) or stay still
        evader_actions[:, 0] = np.clip(evader_actions[:, 0], -self.max_velocity, 0.0)
        
        # Update velocities
        self.pursuer_velocities = pursuer_actions.astype(np.float32)
        self.evader_velocities = evader_actions.astype(np.float32)
        
        # Execute for action_repeat simulation steps
        cumulative_pursuer_reward = 0.0
        cumulative_evader_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self.action_repeat):
            self.steps += 1
            
            # Update positions for active agents
            self.pursuer_positions += self.pursuer_velocities
            
            # Only update evaders that haven't been caught or escaped
            active_evaders = ~(self.evaders_caught | self.evaders_escaped)
            self.evader_positions[active_evaders] += self.evader_velocities[active_evaders]
            
            # Clip positions to world boundaries
            self.pursuer_positions[:, 0] = np.clip(self.pursuer_positions[:, 0], 0, self.world_size_x)
            self.pursuer_positions[:, 1] = np.clip(self.pursuer_positions[:, 1], 0, self.world_size_y)
            self.evader_positions[:, 0] = np.clip(self.evader_positions[:, 0], 0, self.world_size_x)
            self.evader_positions[:, 1] = np.clip(self.evader_positions[:, 1], 0, self.world_size_y)
            
            # Calculate rewards for this simulation step
            pursuer_reward, evader_reward = self._calculate_rewards()
            cumulative_pursuer_reward += pursuer_reward
            cumulative_evader_reward += evader_reward
            
            # Check termination conditions
            all_evaders_done = np.all(self.evaders_caught | self.evaders_escaped)
            terminated = all_evaders_done
            truncated = self.steps >= self.max_steps
            
            # Update previous distances for next step
            self._update_previous_distances()
            
            # Break early if episode ends
            if terminated or truncated:
                break
        
        state = self._get_state()
        info = self._get_info()
        info['action_repeat_steps'] = min(self.action_repeat, self.steps)
        info['pursuer_reward'] = cumulative_pursuer_reward
        info['evader_reward'] = cumulative_evader_reward
        
        # Return combined reward (you can customize this)
        # For multi-agent training, you might want to return both separately
        combined_reward = cumulative_pursuer_reward - cumulative_evader_reward
        
        return state, combined_reward, terminated, truncated, info
    
    def _calculate_rewards(self) -> Tuple[float, float]:
        """Calculate separate rewards for pursuers and evaders."""
        pursuer_reward = 0.0
        evader_reward = 0.0
        
        # Check for captures
        for evader_idx in range(self.n_evaders):
            if not self.evaders_caught[evader_idx] and not self.evaders_escaped[evader_idx]:
                evader_pos = self.evader_positions[evader_idx]
                
                # Check if any pursuer caught this evader
                distances_to_pursuers = np.linalg.norm(
                    self.pursuer_positions - evader_pos,
                    axis=1
                )
                
                if np.any(distances_to_pursuers < self.capture_distance):
                    self.evaders_caught[evader_idx] = True
                    pursuer_reward += self.capture_reward
                    evader_reward -= self.capture_reward
                
                # Check if evader reached escape zone
                elif evader_pos[0] < self.escape_zone_x:
                    self.evaders_escaped[evader_idx] = True
                    evader_reward += self.escape_reward
                    pursuer_reward -= self.escape_reward
        
        # Distance-based rewards for pursuers (encourage chasing)
        for pursuer_idx in range(self.n_pursuers):
            # Find closest uncaught, unescaped evader
            active_evaders = ~(self.evaders_caught | self.evaders_escaped)
            if np.any(active_evaders):
                active_evader_positions = self.evader_positions[active_evaders]
                distances = np.linalg.norm(
                    active_evader_positions - self.pursuer_positions[pursuer_idx],
                    axis=1
                )
                min_distance = np.min(distances)
                
                # Reward for getting closer
                distance_improvement = self.previous_pursuer_distances[pursuer_idx] - min_distance
                pursuer_reward += distance_improvement * self.distance_reward_scale
        
        # Distance-based rewards for evaders (encourage moving to escape zone)
        for evader_idx in range(self.n_evaders):
            if not (self.evaders_caught[evader_idx] or self.evaders_escaped[evader_idx]):
                # Distance to escape zone (left edge)
                distance_to_escape = self.evader_positions[evader_idx, 0] - self.escape_zone_x
                
                # Reward for getting closer to escape
                distance_improvement = self.previous_evader_distances[evader_idx] - distance_to_escape
                evader_reward += distance_improvement * self.distance_reward_scale
        
        # Collision penalties (pursuer-pursuer)
        for i in range(self.n_pursuers):
            for j in range(i + 1, self.n_pursuers):
                distance = np.linalg.norm(
                    self.pursuer_positions[i] - self.pursuer_positions[j]
                )
                if distance < (2 * self.pursuer_radius):
                    pursuer_reward += self.collision_penalty
        
        # Collision penalties (evader-evader)
        active_evaders = ~(self.evaders_caught | self.evaders_escaped)
        active_indices = np.where(active_evaders)[0]
        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                idx_i, idx_j = active_indices[i], active_indices[j]
                distance = np.linalg.norm(
                    self.evader_positions[idx_i] - self.evader_positions[idx_j]
                )
                if distance < (2 * self.evader_radius):
                    evader_reward += self.collision_penalty
        
        # Small time penalty for both teams (encourage efficiency)
        pursuer_reward -= 0.1
        evader_reward -= 0.1
        
        return float(pursuer_reward), float(evader_reward)
    
    def _update_previous_distances(self):
        """Update distance tracking for reward calculation."""
        # Pursuers: track distance to nearest active evader
        for pursuer_idx in range(self.n_pursuers):
            active_evaders = ~(self.evaders_caught | self.evaders_escaped)
            if np.any(active_evaders):
                active_evader_positions = self.evader_positions[active_evaders]
                distances = np.linalg.norm(
                    active_evader_positions - self.pursuer_positions[pursuer_idx],
                    axis=1
                )
                self.previous_pursuer_distances[pursuer_idx] = np.min(distances)
            else:
                self.previous_pursuer_distances[pursuer_idx] = 0.0
        
        # Evaders: track distance to escape zone
        for evader_idx in range(self.n_evaders):
            if not (self.evaders_caught[evader_idx] or self.evaders_escaped[evader_idx]):
                self.previous_evader_distances[evader_idx] = (
                    self.evader_positions[evader_idx, 0] - self.escape_zone_x
                )
            else:
                self.previous_evader_distances[evader_idx] = 0.0
    
    def _get_state(self) -> np.ndarray:
        """Construct the state vector."""
        state = np.concatenate([
            self.pursuer_positions.flatten(),
            self.evader_positions.flatten(),
            self.pursuer_velocities.flatten(),
            self.evader_velocities.flatten(),
            self.evaders_caught.astype(np.float32),
            self.evaders_escaped.astype(np.float32)
        ])
        return state.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Return additional information about the environment state."""
        return {
            'evaders_caught': self.evaders_caught.copy(),
            'evaders_escaped': self.evaders_escaped.copy(),
            'num_caught': np.sum(self.evaders_caught),
            'num_escaped': np.sum(self.evaders_escaped),
            'steps': self.steps,
            'pursuer_positions': self.pursuer_positions.copy(),
            'evader_positions': self.evader_positions.copy(),
        }
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.steps}")
            print(f"Evaders caught: {np.sum(self.evaders_caught)}/{self.n_evaders}")
            print(f"Evaders escaped: {np.sum(self.evaders_escaped)}/{self.n_evaders}")
            print("-" * 50)
    
    def save_animation(self, steps=200, filename=None):
        """
        Run the environment with random actions and save as GIF.
        
        Args:
            steps: Number of ACTION steps to simulate
            filename: Optional filename (default: auto-generated with timestamp)
        """
        import matplotlib.pyplot as plt
        import imageio
        import matplotlib as mpl
        from datetime import datetime
        import os
        
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        animations_dir = os.path.join(script_dir, 'Animations')
        os.makedirs(animations_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'adversarial_chase_{timestamp}.gif'
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{filename}_adversarial_{timestamp}.gif'
        
        save_path = os.path.join(animations_dir, filename)
        
        # Reset environment
        self.reset()
        
        frames = []

        # Global style
        mpl.rcParams.update({
            "axes.facecolor": "#f8f9fa",
            "figure.facecolor": "#f8f9fa",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.edgecolor": "#cccccc",
            "font.family": "DejaVu Sans",
            "font.size": 11,
        })

        print(f"🎞️ Generating {steps} frames (action_repeat={self.action_repeat})...")

        for step in range(steps):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_xlim(0, self.world_size_x)
            ax.set_ylim(0, self.world_size_y)
            ax.set_aspect('equal')
            ax.set_title(f"Step {self.steps} | Caught: {np.sum(self.evaders_caught)} | Escaped: {np.sum(self.evaders_escaped)}", 
                        fontsize=13, fontweight="bold", color="#333")

            # Draw escape zone
            ax.axvspan(0, self.escape_zone_x, alpha=0.2, color='green', label='Escape Zone')

            # --- Evaders ---
            active_evaders = ~(self.evaders_caught | self.evaders_escaped)
            caught_evaders = self.evaders_caught
            escaped_evaders = self.evaders_escaped
            
            # Active evaders (red)
            if np.any(active_evaders):
                active_pos = self.evader_positions[active_evaders]
                ax.scatter(active_pos[:, 0], active_pos[:, 1],
                          s=200, c='#e63946', marker='s', edgecolors='white', 
                          linewidths=1.5, label='Evaders', zorder=4)
                
                # Velocity arrows for active evaders
                active_vel = self.evader_velocities[active_evaders]
                for pos, vel in zip(active_pos, active_vel):
                    if np.linalg.norm(vel) > 1e-6:
                        ax.arrow(pos[0], pos[1], vel[0]*2, vel[1]*2,
                                head_width=0.25, head_length=0.2,
                                fc='#c1121f', ec='#9b2226', alpha=0.8, zorder=5)
            
            # Caught evaders (gray)
            if np.any(caught_evaders):
                caught_pos = self.evader_positions[caught_evaders]
                ax.scatter(caught_pos[:, 0], caught_pos[:, 1],
                          s=200, c='#6c757d', marker='s', alpha=0.3, zorder=2)
            
            # Escaped evaders (green)
            if np.any(escaped_evaders):
                escaped_pos = self.evader_positions[escaped_evaders]
                ax.scatter(escaped_pos[:, 0], escaped_pos[:, 1],
                          s=200, c='#52b788', marker='s', alpha=0.6, zorder=3)

            # --- Pursuers (blue) ---
            ax.scatter(self.pursuer_positions[:, 0], self.pursuer_positions[:, 1],
                      s=200, c='#1d4ed8', marker='o', edgecolors='white', 
                      linewidths=1.5, label='Pursuers', zorder=4)
            
            # Velocity arrows for pursuers
            for pos, vel in zip(self.pursuer_positions, self.pursuer_velocities):
                if np.linalg.norm(vel) > 1e-6:
                    ax.arrow(pos[0], pos[1], vel[0]*2, vel[1]*2,
                            head_width=0.25, head_length=0.2,
                            fc='#2563eb', ec='#1e40af', alpha=0.8, zorder=5)

            ax.legend(loc='upper right', framealpha=0.9)

            # Convert to frame
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(frame)
            plt.close(fig)

            # Step simulation
            action = self.action_space.sample()
            _, _, terminated, truncated, _ = self.step(action)

            if terminated or truncated:
                print(f"🚩 Episode ended at step {step+1} (sim step {self.steps})")
                frames.extend([frames[-1]] * 10)  # Pause at end
                break

            if (step + 1) % 50 == 0:
                print(f"  ✅ {step + 1}/{steps} action steps")

        # Save GIF
        print(f"💾 Saving animation to {save_path} ...")
        imageio.mimsave(save_path, frames, duration=0.06)
        print(f"✅ Done! Total frames: {len(frames)}  |  Duration: {len(frames)*0.06:.1f}s")

        return save_path


# Example usage
if __name__ == "__main__":
    # Create adversarial environment
    action_repeat = 3
    env = AdversarialChaseEnv(
        n_pursuers=3,
        n_evaders=3,
        max_velocity=0.5,
        capture_distance=0.8,
        escape_zone_x=3.0,
        action_repeat=action_repeat
    )
    
    # Test environment
    state, info = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action repeat: {action_repeat}")
    
    # Run a few random steps
    print("\nTesting adversarial environment:")
    for i in range(5):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}:")
        print(f"  Combined reward: {reward:.2f}")
        print(f"  Pursuer reward: {info['pursuer_reward']:.2f}")
        print(f"  Evader reward: {info['evader_reward']:.2f}")
        print(f"  Caught: {info['num_caught']}, Escaped: {info['num_escaped']}")
        
        if terminated or truncated:
            break
    
    print("\n✓ Adversarial environment created successfully!")
    print(f"✓ Pursuers try to catch evaders")
    print(f"✓ Evaders try to reach escape zone (left {env.escape_zone_x} units)")
    print(f"✓ Evaders can only move left (x-velocity <= 0)")
    print(f"✓ Separate rewards for each team")
    
    # Generate and save animation
    print("\n" + "="*50)
    print("Generating animation...")
    print("="*50)
    env.save_animation(steps=100)