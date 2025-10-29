import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Vars
GAMMA = 0.99
LAMBDA = 0.95

# ----------------------------------------------------------
# Standalone Visualization Function
# ----------------------------------------------------------
def create_animation_from_observations(observations_list, width_ratio=3.0, save_path="episode_animation.gif"):
    """
    Create an animated visualization from a list of raw observations

    Args:
        observations_list: List of observation dictionaries from the environment
            Each observation dict has format: {agent_name: obs_array, ...}
            Example: [{'good_0': array([...]), 'adversary_0': array([...])}, ...]
        width_ratio: Width of the environment (for axis limits)
        save_path: Path to save the animation GIF

    The function extracts agent positions from the observations and creates animation.
    Observations are expected to have format: [own_pos(2), own_vel(2), nearest_adv(2),
    nearest_good(2), adv_box(4), good_box(4)]
    """

    # Extract positions from observations
    trajectory = []
    cumulative_reward = 0  # You can pass rewards separately if needed

    for step, obs_dict in enumerate(observations_list):
        good_pos = []
        adv_pos = []
        good_active = []
        adv_active = []

        for agent_name, obs in obs_dict.items():
            # Extract own position (first 2 elements of observation)
            pos = obs[:2]
            is_active = not np.allclose(obs, 0)  # Check if obs is all zeros (inactive)

            if 'good' in agent_name:
                good_pos.append(pos)
                good_active.append(is_active)
            elif 'adversary' in agent_name:
                adv_pos.append(pos)
                adv_active.append(is_active)

        state = {
            'good_pos': good_pos,
            'adv_pos': adv_pos,
            'good_active': good_active,
            'adv_active': adv_active,
            'step': step,
            'reward': cumulative_reward
        }
        trajectory.append(state)

    # Create animation
    fig, ax = plt.subplots(figsize=(12, 6))

    def animate(frame):
        ax.clear()
        state = trajectory[frame]

        # Plot good agents (blue)
        if state['good_pos']:
            good_pos = np.array(state['good_pos'])
            good_active = np.array(state['good_active'])

            if good_active.any():
                ax.scatter(good_pos[good_active, 0], good_pos[good_active, 1],
                           c='blue', s=200, label='Good Agents', edgecolors='black', linewidths=2)
            # if (~good_active).any():
            #     ax.scatter(good_pos[~good_active, 0], good_pos[~good_active, 1],
            #                c='lightblue', s=200, alpha=0.3, edgecolors='black', linewidths=1)

        # Plot adversaries (red)
        if state['adv_pos']:
            adv_pos = np.array(state['adv_pos'])
            adv_active = np.array(state['adv_active'])

            if adv_active.any():
                ax.scatter(adv_pos[adv_active, 0], adv_pos[adv_active, 1],
                           c='red', s=200, label='Adversaries', marker='s', edgecolors='black', linewidths=2)
            # if (~adv_active).any():
            #     ax.scatter(adv_pos[~adv_active, 0], adv_pos[~adv_active, 1],
            #                c='lightcoral', s=200, alpha=0.3, marker='s', edgecolors='black', linewidths=1)

        # Goal line for adversaries
        ax.axvline(x=-width_ratio + 0.1, color='green', linestyle='--', linewidth=2, label='Goal Line')

        ax.set_xlim(-width_ratio, width_ratio)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(f'Step {state["step"]}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        # ax.set_aspect('equal')

    anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=50, repeat=True)
    anim.save(save_path, writer='pillow', fps=20)
    plt.close()

    print(f"✅ Animation saved to {save_path}")

# ----------------------------------------------------------
# PPO Actor-Critic Network
# ----------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def act(self, obs):
        mean = self.actor(obs)
        dist = Normal(mean, torch.ones_like(mean) * 0.1)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1, 1), log_prob

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        dist = Normal(mean, torch.ones_like(mean) * 0.1)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

# ----------------------------------------------------------
# Helper: Compute GAE
# ----------------------------------------------------------
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages), torch.tensor(returns)