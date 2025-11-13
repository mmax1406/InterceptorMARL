import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Optional


class MultiAgentTargetEnv(gym.Env):
    """
    Multi-agent reinforcement learning environment where N agents must reach N targets.
    
    Features:
    - Agents start on left half, targets on right half
    - Continuous action space (velocity vectors)
    - Rewards based on distance to targets and collision avoidance
    - Episode ends when all targets are reached
    - Supports cooperative training
    - Optional disturbance system (currently disabled)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        n_agents: int = 3,
        n_targets: int = 3,
        world_size: float = 10.0,
        max_velocity: float = 0.5,
        agent_radius: float = 0.2,
        target_radius: float = 0.3,
        collision_penalty: float = -10.0,
        target_reward: float = 100.0,
        distance_reward_scale: float = 1.0,
        collision_distance_scale: float = 2.0,
        max_steps: int = 500,
        apply_disturbances: bool = False,
        disturbance_strength: float = 0.1,
        disturbance_frequency: float = 0.1,
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.world_size = world_size
        self.max_velocity = max_velocity
        self.agent_radius = agent_radius
        self.target_radius = target_radius
        self.collision_penalty = collision_penalty
        self.target_reward = target_reward
        self.distance_reward_scale = distance_reward_scale
        self.collision_distance_scale = collision_distance_scale
        self.max_steps = max_steps
        self.apply_disturbances = apply_disturbances
        self.disturbance_strength = disturbance_strength
        self.disturbance_frequency = disturbance_frequency
        
        # State space: [agent_positions (2*N), target_positions (2*N), 
        #                agent_velocities (2*N), targets_reached (N)]
        state_dim = 2 * n_agents + 2 * n_targets + 2 * n_agents + n_targets
        self.observation_space = spaces.Box(
            low=-world_size,
            high=world_size,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: velocity vector (vx, vy) for each agent
        self.action_space = spaces.Box(
            low=-max_velocity,
            high=max_velocity,
            shape=(n_agents * 2,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.agent_positions = np.zeros((n_agents, 2), dtype=np.float32)
        self.agent_velocities = np.zeros((n_agents, 2), dtype=np.float32)
        self.target_positions = np.zeros((n_targets, 2), dtype=np.float32)
        self.targets_reached = np.zeros(n_targets, dtype=bool)
        self.steps = 0
        self.previous_min_distances = np.zeros(n_agents, dtype=np.float32)
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset agents on left half of screen
        self.agent_positions = np.random.uniform(
            low=[0, 0],
            high=[self.world_size / 2, self.world_size],
            size=(self.n_agents, 2)
        ).astype(np.float32)
        
        # Reset targets on right half of screen
        self.target_positions = np.random.uniform(
            low=[self.world_size / 2, 0],
            high=[self.world_size, self.world_size],
            size=(self.n_targets, 2)
        ).astype(np.float32)
        
        # Reset velocities and targets reached
        self.agent_velocities = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.targets_reached = np.zeros(self.n_targets, dtype=bool)
        self.steps = 0
        
        # Calculate initial minimum distances for each agent
        self._update_previous_distances()
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Array of shape (n_agents * 2,) containing velocity vectors
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1
        
        # Reshape action to (n_agents, 2)
        actions = action.reshape(self.n_agents, 2)
        
        # Clip actions to max velocity
        actions = np.clip(actions, -self.max_velocity, self.max_velocity)
        
        # Update velocities
        self.agent_velocities = actions.astype(np.float32)
        
        # Apply disturbances if enabled
        if self.apply_disturbances and np.random.random() < self.disturbance_frequency:
            disturbance = np.random.uniform(
                -self.disturbance_strength,
                self.disturbance_strength,
                size=(self.n_agents, 2)
            )
            self.agent_velocities += disturbance
        
        # Update positions
        self.agent_positions += self.agent_velocities
        
        # Clip positions to world boundaries
        self.agent_positions = np.clip(
            self.agent_positions,
            0,
            self.world_size
        )
        
        # Calculate rewards
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = np.all(self.targets_reached)
        truncated = self.steps >= self.max_steps
        
        # Update previous distances for next step
        self._update_previous_distances()
        
        state = self._get_state()
        info = self._get_info()
        
        return state, reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on distances, collisions, and target reaching."""
        total_reward = 0.0
        
        # Check for newly reached targets
        for i, target_pos in enumerate(self.target_positions):
            if not self.targets_reached[i]:
                # Calculate distances from all agents to this target
                distances = np.linalg.norm(
                    self.agent_positions - target_pos,
                    axis=1
                )
                
                # Check if any agent reached the target
                if np.any(distances < (self.agent_radius + self.target_radius)):
                    self.targets_reached[i] = True
                    # Global reward for reaching a target
                    total_reward += self.target_reward
        
        # Distance-based rewards (encourage moving closer to targets)
        for agent_idx in range(self.n_agents):
            # Find closest unreached target
            unreached_mask = ~self.targets_reached
            if np.any(unreached_mask):
                unreached_targets = self.target_positions[unreached_mask]
                distances = np.linalg.norm(
                    unreached_targets - self.agent_positions[agent_idx],
                    axis=1
                )
                min_distance = np.min(distances)
                
                # Reward for getting closer to nearest target
                distance_improvement = self.previous_min_distances[agent_idx] - min_distance
                total_reward += distance_improvement * self.distance_reward_scale
        
        # Collision penalties (agent-agent collisions)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                distance = np.linalg.norm(
                    self.agent_positions[i] - self.agent_positions[j]
                )
                
                # Collision detected
                if distance < (2 * self.agent_radius):
                    total_reward += self.collision_penalty
                
                # Soft penalty for being too close
                elif distance < (self.collision_distance_scale * self.agent_radius):
                    proximity_penalty = -1.0 * (1.0 - distance / (self.collision_distance_scale * self.agent_radius))
                    total_reward += proximity_penalty
        
        # Small penalty for each step (encourage efficiency)
        total_reward -= 0.1
        
        return float(total_reward)
    
    def _update_previous_distances(self):
        """Update the minimum distance from each agent to unreached targets."""
        for agent_idx in range(self.n_agents):
            unreached_mask = ~self.targets_reached
            if np.any(unreached_mask):
                unreached_targets = self.target_positions[unreached_mask]
                distances = np.linalg.norm(
                    unreached_targets - self.agent_positions[agent_idx],
                    axis=1
                )
                self.previous_min_distances[agent_idx] = np.min(distances)
            else:
                self.previous_min_distances[agent_idx] = 0.0
    
    def _get_state(self) -> np.ndarray:
        """Construct the state vector."""
        state = np.concatenate([
            self.agent_positions.flatten(),
            self.target_positions.flatten(),
            self.agent_velocities.flatten(),
            self.targets_reached.astype(np.float32)
        ])
        return state.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Return additional information about the environment state."""
        return {
            'targets_reached': self.targets_reached.copy(),
            'num_targets_reached': np.sum(self.targets_reached),
            'steps': self.steps,
            'agent_positions': self.agent_positions.copy(),
            'target_positions': self.target_positions.copy(),
        }
    
    def render(self, mode: str = 'human'):
        """Render the environment (placeholder for visualization)."""
        if mode == 'human':
            print(f"Step: {self.steps}")
            print(f"Targets reached: {np.sum(self.targets_reached)}/{self.n_targets}")
            print(f"Agent positions:\n{self.agent_positions}")
            print(f"Target positions:\n{self.target_positions}")
            print("-" * 50)
    
    def render_frame(self, ax):
        """Render a single frame for animation."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        ax.clear()
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect('equal')
        ax.set_title(f'Step: {self.steps} | Targets: {np.sum(self.targets_reached)}/{self.n_targets}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        # Draw left/right halves background
        ax.axvline(x=self.world_size/2, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.fill_between([0, self.world_size/2], 0, self.world_size, alpha=0.05, color='blue', label='Agent spawn zone')
        ax.fill_between([self.world_size/2, self.world_size], 0, self.world_size, alpha=0.05, color='red', label='Target zone')
        
        # Draw agents
        for i, pos in enumerate(self.agent_positions):
            circle = Circle(pos, self.agent_radius, color='blue', alpha=0.7, label='Agent' if i == 0 else '')
            ax.add_patch(circle)
            # Draw velocity vector
            vel = self.agent_velocities[i]
            if np.linalg.norm(vel) > 1e-6:
                ax.arrow(pos[0], pos[1], vel[0]*2, vel[1]*2, 
                        head_width=0.15, head_length=0.1, fc='darkblue', ec='darkblue', alpha=0.6)
            ax.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        
        # Draw targets
        for i, pos in enumerate(self.target_positions):
            if self.targets_reached[i]:
                # Reached targets (faded)
                circle = Circle(pos, self.target_radius, color='green', alpha=0.3, linestyle='--', fill=False)
                ax.add_patch(circle)
            else:
                # Unreached targets
                circle = Circle(pos, self.target_radius, color='red', alpha=0.7, label='Target' if i == 0 else '')
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    def save_animation(self, steps=200, filename=None):
        """
        Run the environment with random actions and save as GIF.
        
        Args:
            steps: Number of steps to simulate
            filename: Optional filename (default: auto-generated with timestamp)
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from datetime import datetime
        import os
        
        # Create animations folder if it doesn't exist
        os.makedirs('animations', exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'animations/multi_agent_env_{timestamp}.gif'
        elif not filename.startswith('animations/'):
            filename = f'animations/{filename}'
        
        # Reset environment
        self.reset()
        
        # Create figure with fixed size
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        frames = []
        
        # Simulate and store frames
        print(f"Generating {steps} frames...")
        for step in range(steps):
            # Render current frame
            self.render_frame(ax)
            
            # Convert plot to image using buffer_rgba instead of tostring_rgb
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)
            # Convert RGBA to RGB
            image = image[:, :, :3]
            frames.append(image)
            
            # Take random action
            action = self.action_space.sample()
            _, _, terminated, truncated, _ = self.step(action)
            
            if terminated or truncated:
                print(f"Episode ended at step {step+1}")
                # Add a few frames at the end
                for _ in range(10):
                    self.render_frame(ax)
                    fig.canvas.draw()
                    buf = fig.canvas.buffer_rgba()
                    image = np.asarray(buf)
                    image = image[:, :, :3]
                    frames.append(image)
                break
            
            if (step + 1) % 50 == 0:
                print(f"  {step + 1}/{steps} frames generated")
        
        plt.close(fig)
        
        # Save as GIF using matplotlib animation
        print(f"Saving animation to {filename}...")
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        im = ax.imshow(frames[0])
        ax.axis('off')
        
        def update_frame(frame_idx):
            im.set_array(frames[frame_idx])
            return [im]
        
        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(frames), 
            interval=50, blit=True, repeat=True
        )
        
        anim.save(filename, writer='pillow', fps=20)
        plt.close(fig)
        
        print(f"✓ Animation saved to {filename}")
        print(f"  Total frames: {len(frames)}")
        print(f"  Duration: {len(frames)/20:.1f} seconds")
        
        return filename


# Example usage
if __name__ == "__main__":
    # Create environment
    env = MultiAgentTargetEnv(
        n_agents=3,
        n_targets=3,
        world_size=10.0,
        max_velocity=0.5,
        apply_disturbances=False  # Disturbances disabled by default
    )
    
    # Test environment
    state, info = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few random steps
    for _ in range(5):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.2f}, Terminated: {terminated}, Targets: {info['num_targets_reached']}/{env.n_targets}")
        
        if terminated or truncated:
            break
    
    print("\n✓ Environment created successfully!")
    print(f"✓ Compatible with AgileRL (Gymnasium interface)")
    print(f"✓ Disturbance system ready (currently disabled)")
    
    # Generate and save animation
    print("\n" + "="*50)
    print("Generating animation...")
    print("="*50)
    env.save_animation(steps=200)