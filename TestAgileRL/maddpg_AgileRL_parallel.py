import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer

# ------------------------ Utilities ------------------------

def render_static_frames(data, save_path="./Animations/testAlgo_AgileRL_parallel.gif"):
    """Render recorded positions with a fixed camera."""
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

# ------------------------ Helpers for vectorized envs ------------------------

def safe_reset(env):
    """Reset a PettingZoo parallel env and return the obs dict (compat with different returns)."""
    out = env.reset()
    # Some versions return (obs, infos) when render_mode is set; support either.
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out

# ------------------------ Main Script ------------------------

if __name__ == '__main__':
    try:
        from pettingzoo.mpe import simple_spread_v3
        print("PettingZoo imported successfully.")
    except ImportError:
        raise ImportError("PettingZoo is not installed. Install with: pip install pettingzoo")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, plot = True, True

    # --- Multi-env (vectorized) settings ---
    NUM_ENVS = 8                     # <<-- number of parallel environments (instances)
    max_steps = 25
    num_agents = 3                   # keep consistent with env N=3 below
    envs = []
    for i in range(NUM_ENVS):
        e = simple_spread_v3.parallel_env(N=num_agents, local_ratio=0.5, max_cycles=max_steps, continuous_actions=True)
        envs.append(e)

    single_env = simple_spread_v3.parallel_env(N=num_agents, local_ratio=0.5, max_cycles=max_steps, continuous_actions=True) #For evaluation metric

    # Reset all envs and check agent list
    obs_list = [safe_reset(e) for e in envs]
    agent_ids = envs[0].agents[:]   # order should be same for all envs

    # Get state and action dims from first env agent
    state_dims = [envs[0].observation_space(agent).shape for agent in agent_ids]
    action_dims = [envs[0].action_space(agent).shape[0] for agent in agent_ids]

    discrete_actions = False
    min_action = envs[0].action_space(agent_ids[0]).low[0]
    max_action = envs[0].action_space(agent_ids[0]).high[0]

    # --- Initialize AgileRL MADDPG ---
    maddpg = MADDPG(
        state_dims=state_dims,
        action_dims=action_dims,
        one_hot=False,
        n_agents=num_agents,
        agent_ids=agent_ids,
        max_action=[[max_action] * action_dims[i] for i in range(num_agents)],
        min_action=[[min_action] * action_dims[i] for i in range(num_agents)],
        discrete_actions=discrete_actions,
        net_config={
            'arch': 'mlp',
            'h_size': [256, 256]
        },
        batch_size=1024,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01,
        expl_noise=0.1,
        device=device,
    )

    # --- Initialize Replay Buffer ---
    memory = MultiAgentReplayBuffer(
        memory_size=1_000_000,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agent_ids,
        device=device,
    )

    save_dir = "checkpoints/Weights_Parallel.pt"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    if train:
        # --- Logging setup ---
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "training_log_parallel.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "avg_agent_reward"])

        num_episodes = 5_000
        print_every = 100
        best_reward = -999999
        update_every = 100    # perform learning once every X global steps (across envs)
        total_steps = 0

        for ep in range(1, num_episodes + 1):
            # Reset all envs for a new "batch" of episodes
            obs_list = [safe_reset(e) for e in envs]
            # per-env agent cumulative rewards (shape: NUM_ENVS x n_agents)
            ep_rewards_per_env = np.zeros((NUM_ENVS, num_agents))

            # Run all environments synchronously for up to max_steps
            for step in range(max_steps):
                total_steps += 1

                # For each env: build the obs_dict and get actions, then step
                for env_idx, env in enumerate(envs):
                    obs = obs_list[env_idx]
                    # Build obs dict expected by MADDPG
                    obs_dict = {agent_id: obs[agent_id] for agent_id in agent_ids}

                    # getAction returns continuous actions dict for this environment
                    cont_actions, raw_action = maddpg.getAction(obs_dict, epsilon=0.5)

                    # step and collect
                    actions_dict = {agent_id: cont_actions[agent_id] for agent_id in agent_ids}
                    next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)

                    # Compute dones (per-agent)
                    dones = {agent: terminations[agent] or truncations[agent] for agent in agent_ids}

                    # Save transition to replay buffer
                    memory.save2memory(obs, cont_actions, rewards, next_obs, dones)

                    # accumulate rewards
                    ep_rewards_per_env[env_idx] += np.array([rewards[agent] for agent in agent_ids])

                    # update local obs
                    obs_list[env_idx] = next_obs

                    # If env finished all agents, we could early-reset it (optional)
                    if all(dones.values()) or len(env.agents) == 0:
                        # reset this env early so it continues producing data for next steps
                        obs_list[env_idx] = safe_reset(env)

                # Perform learning when we have enough samples and at desired frequency
                if len(memory) >= maddpg.batch_size and total_steps % update_every == 0:
                    experiences = memory.sample(maddpg.batch_size)
                    maddpg.learn(experiences)

            # Evaluate model properly
            fitness_mean, fitness_std = maddpg.test(single_env, max_steps=max_steps, loop=10)

            # After running max_steps across NUM_ENVS, compute episode rewards
            # We summarize by summing across envs (treat each env as separate episode)
            total_rewards_each_env = ep_rewards_per_env.sum(axis=1)     # sum per env across agents
            avg_agent_reward_each_env = ep_rewards_per_env.mean(axis=1) # avg per-agent per env

            # We'll log the mean across the NUM_ENVS batch to track progress
            mean_total_reward = float(total_rewards_each_env.mean())
            mean_avg_agent_reward = float(avg_agent_reward_each_env.mean())

            # Save best model (by mean_total_reward)
            # if mean_total_reward > best_reward:
            if fitness_mean > best_reward:
                best_reward = fitness_mean
                maddpg.saveCheckpoint(save_dir)
                print(f"[Episode {ep}] 🏆 New best model saved! Mean total reward: {best_reward:.2f}")

            # CSV logging
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep, mean_total_reward, mean_avg_agent_reward, fitness_mean, fitness_std])

            if ep % print_every == 0:
                print(f"[Episode {ep}] Mean Total: {mean_total_reward:.2f}, Mean per-agent: {mean_avg_agent_reward:.2f}")

        print("Training finished ✅")

    # --- Evaluation / Visualization (single env for clarity) ---
    if plot:
        eval_env = simple_spread_v3.parallel_env(
            N=num_agents, local_ratio=0.5, max_cycles=max_steps,
            continuous_actions=True, render_mode="rgb_array"
        )
        maddpg.loadCheckpoint(save_dir)

        with torch.no_grad():
            obs = safe_reset(eval_env)
            ep_rewards = np.zeros(num_agents)
            data = []

            for step in range(max_steps):
                obs_dict = {agent_id: obs[agent_id] for agent_id in agent_ids}
                cont_actions, raw_action = maddpg.getAction(obs_dict, epsilon=0.0)
                actions_dict = {agent_id: cont_actions[agent_id] for agent_id in agent_ids}
                next_obs, rewards, terminations, truncations, infos = eval_env.step(actions_dict)
                dones = {agent: terminations[agent] or truncations[agent] for agent in agent_ids}

                # Record positions (works for simple_spread internals)
                agent_positions = [agent.state.p_pos.copy() for agent in eval_env.unwrapped.world.agents]
                landmark_positions = [lm.state.p_pos.copy() for lm in eval_env.unwrapped.world.landmarks]
                data.append({
                    "step": step,
                    "agents": agent_positions,
                    "landmarks": landmark_positions
                })

                obs = next_obs
                ep_rewards += np.array([rewards[agent] for agent in agent_ids])
                if len(eval_env.agents) == 0 or all(dones.values()):
                    break

        gif_path = "./Animations/"
        os.makedirs(gif_path, exist_ok=True)
        render_static_frames(data)

    # Close all envs
    for e in envs:
        e.close()
