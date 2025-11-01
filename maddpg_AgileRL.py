import os
import csv
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imageio
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer

# ------------------------ Utilities ------------------------

def plotPrettyLog(data, window_size=20):
    """Plot training rewards with rolling mean and std."""
    series = pd.Series(data)
    rolling_mean = series.rolling(window_size).mean()
    rolling_std = series.rolling(window_size).std()
    episodes = np.arange(len(data))

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(x=episodes, y=data, color='lightgray', alpha=0.4, label='Raw Rewards')
    sns.lineplot(x=episodes, y=rolling_mean, color='blue', label=f'Mean (window={window_size})')
    plt.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     color='blue', alpha=0.2, label='±1 Std. Dev.')
    plt.title("Reward Progress with Sliding Window")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('training_reward_agileRL.png')
    plt.show(block=False)

def render_static_frames(data, save_path="./Animations/testAlgo_AgileRL.gif"):
    """Render all recorded positions with a fixed camera."""
    frames = []
    cam_range = 3

    for record in data:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-cam_range, cam_range)
        ax.set_ylim(-cam_range, cam_range)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        landmarks = np.array(record["landmarks"])
        if len(landmarks) > 0:
            ax.scatter(landmarks[:, 0], landmarks[:, 1], s=150, c="gray", label="Landmarks")

        agents = np.array(record["agents"])
        ax.scatter(agents[:, 0], agents[:, 1], s=150, c="tab:blue", label="Agents")
        ax.text(-cam_range + 0.05, cam_range - 0.1, f"Step {record['step']}", fontsize=10, color='black')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.1)
    print(f"✅ Saved static GIF at: {save_path}")

# ------------------------ Main Script ------------------------

if __name__ == '__main__':
    try:
        from pettingzoo.mpe import simple_spread_v3

        print("PettingZoo imported successfully.")
    except ImportError:
        raise ImportError("PettingZoo is not installed. Install with: pip install pettingzoo")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, plot = False, True

    # --- Initialize Environment ---
    max_steps = 25
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=max_steps, continuous_actions=True)
    env.reset()

    num_agents = len(env.agents)
    agent_ids = env.agents

    # Get state and action dimensions for each agent
    state_dims = [env.observation_space(agent).shape for agent in agent_ids]
    action_dims = [env.action_space(agent).shape[0] for agent in agent_ids]

    # AgileRL expects one-hot discrete actions, but we have continuous
    # For continuous actions, discrete_actions should be False
    discrete_actions = False

    # Get min/max action values
    min_action = env.action_space(agent_ids[0]).low[0]
    max_action = env.action_space(agent_ids[0]).high[0]

    # --- Initialize AgileRL MADDPG ---
    maddpg = MADDPG(
        state_dims=state_dims,
        action_dims=action_dims,
        one_hot=False,  # Not using one-hot encoding for continuous actions
        n_agents=num_agents,
        agent_ids=agent_ids,
        max_action=[[max_action] * action_dims[i] for i in range(num_agents)],
        min_action=[[min_action] * action_dims[i] for i in range(num_agents)],
        discrete_actions=discrete_actions,

        # Network architecture
        net_config={
            'arch': 'mlp',
            'h_size': [256, 256]  # Actor hidden layers
        },

        # Hyperparameters
        batch_size=1024,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01,
        # policy_freq=1,  # Update policy every step (like updates_per_step=1)
        expl_noise=0.1,  # Equivalent to OU sigma
        device=device,
    )

    # --- Initialize Replay Buffer ---
    memory = MultiAgentReplayBuffer(
        memory_size=1_000_000,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agent_ids,
        device=device,
    )

    save_dir = "checkpoints_agilerl/test.pt"
    # os.makedirs(save_dir, exist_ok=True)

    if train:
        # --- Logging setup ---
        log_dir = "logs_agilerl"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "training_log.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "avg_agent_reward"])

        # --- Training loop ---
        num_episodes = 200_000
        print_every = 100
        best_reward = -999999
        reward_list = []
        update_every = 100
        total_steps = 0

        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            ep_rewards = np.zeros(num_agents)

            for step in range(max_steps):
                total_steps += 1

                # Get actions from MADDPG (AgileRL format)
                # AgileRL expects dict of observations
                obs_dict = {agent_id: obs[agent_id] for agent_id in agent_ids}

                # Get continuous actions with exploration noise
                cont_actions, raw_action =  maddpg.getAction(obs_dict, epsilon=1.0)

                # Execute actions in environment
                actions_dict = {agent_id: cont_actions[agent_id] for agent_id in agent_ids}
                next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)

                # Process dones
                dones = {agent: terminations[agent] or truncations[agent] for agent in agent_ids}

                # Store in replay buffer
                memory.save2memory(
                    obs, cont_actions, rewards, next_obs, dones
                )

                # Update networks
                if len(memory) >= maddpg.batch_size and total_steps % update_every == 0:
                    experiences = memory.sample(maddpg.batch_size)
                    maddpg.learn(experiences)

                obs = next_obs
                ep_rewards += np.array([rewards[agent] for agent in agent_ids])

                if all(dones.values()) or len(env.agents) == 0:
                    break

            total_reward = np.sum(ep_rewards)
            avg_agent_reward = np.mean(ep_rewards)
            reward_list.append(total_reward)

            # --- Save best model ---
            if total_reward > best_reward:
                best_reward = total_reward
                maddpg.saveCheckpoint(save_dir)
                print(f"[Episode {ep}] 🏆 New best model saved! Total reward: {best_reward:.2f}")

            # --- CSV logging ---
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep, total_reward, avg_agent_reward])

            if ep % print_every == 0:
                print(f"[Episode {ep}] Total: {total_reward:.2f}, Avg per agent: {avg_agent_reward:.2f}")

        plotPrettyLog(reward_list)
        print("Training finished ✅")

    # --- Evaluation/Visualization ---
    if plot:
        env = simple_spread_v3.parallel_env(
            N=3, local_ratio=0.5, max_cycles=max_steps,
            continuous_actions=True, render_mode="rgb_array"
        )

        # Load best model
        maddpg.loadCheckpoint(save_dir)

        with torch.no_grad():
            obs, _ = env.reset()
            ep_rewards = np.zeros(num_agents)
            data = []

            for step in range(max_steps):
                obs_dict = {agent_id: obs[agent_id] for agent_id in agent_ids}

                # Get actions without exploration noise
                cont_actions, raw_action =  maddpg.getAction(obs_dict, epsilon=1.0)

                actions_dict = {agent_id: cont_actions[agent_id] for agent_id in agent_ids}
                next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
                dones = {agent: terminations[agent] or truncations[agent] for agent in agent_ids}

                # Record positions
                agent_positions = [agent.state.p_pos.copy() for agent in env.unwrapped.world.agents]
                landmark_positions = [lm.state.p_pos.copy() for lm in env.unwrapped.world.landmarks]
                data.append({
                    "step": step,
                    "agents": agent_positions,
                    "landmarks": landmark_positions
                })

                obs = next_obs
                if len(env.agents) == 0 or all(dones.values()):
                    break

        gif_path = "./Animations/"
        os.makedirs(gif_path, exist_ok=True)
        render_static_frames(data)

    env.close()