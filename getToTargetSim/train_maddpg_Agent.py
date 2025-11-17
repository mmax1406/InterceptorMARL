import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from datetime import datetime
import sim_spread
from tqdm import tqdm
import burnin_policy
from copy import copy

# ------------------------ Helpers ------------------------

def safe_reset(env):
    """Reset a PettingZoo parallel env and return the obs dict (compat with different returns)."""
    out = env.reset()
    # Some versions return (obs, infos) when render_mode is set; support either.
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out

def get_expl_noise(episode, total_episodes, start=0.3, end=0.02, decay_fraction=0.8):
    decay_episodes = int(total_episodes * decay_fraction)
    if episode >= decay_episodes:
        return end
    else:
        # Linear decay
        return start - (start - end) * (episode / decay_episodes)

# ------------------------ Main Script ------------------------

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Multi-env (vectorized) settings ---
    NUM_ENVS = 32                     
    max_steps = 50
    num_agents = 3                   
    envs = []
    max_vel = 0.1
    for i in range(NUM_ENVS):
        e = sim_spread.MultiAgentTargetEnv( n_agents=num_agents, n_targets=num_agents, max_steps=max_steps, max_velocity=max_vel, apply_disturbances=False)
        envs.append(e)

    single_env = copy(envs[0])

    # Reset all envs and check agent list
    obs_list = [safe_reset(e) for e in envs]
    agent_ids = envs[0].agents[:]   # order should be same for all envs

    # Get state and action dims from first env agent
    state_dims = [envs[0].observation_space(agent).shape for agent in agent_ids]
    action_dims = [envs[0].action_space(agent).shape[0] for agent in agent_ids]

    discrete_actions = False
    min_action = envs[0].action_space(agent_ids[0]).low[0]
    max_action = envs[0].action_space(agent_ids[0]).high[0]

    # Training values
    num_episodes = 2_000
    plot_every = 20
    best_reward = -999999
    update_every = 50    # perform learning once every X global steps (across envs)
    total_steps = 0
    saveAnimation = 50
    noise = max_vel/10

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
            'h_size': [128, 128]
        },
        batch_size=512,
        lr_actor=0.0005,
        lr_critic=0.0005,
        gamma=0.97,
        tau=0.017,
        expl_noise=noise,
        device=device,
    )

    # --- Initialize Replay Buffer ---
    memory = MultiAgentReplayBuffer(
        memory_size=1_000_000,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agent_ids,
        device=device,
    )

    # --- Weights save dir ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    # Generate filename if not provided
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'maddpg_agent_weights_{timestamp}.pt'
    save_dir = os.path.join(save_dir, filename)

    # --- Logging csv setup ---
    log_dir = os.path.join(script_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    filename = f'maddpg_training_log_{timestamp}.csv'
    csv_path = os.path.join(log_dir, filename)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "avg_agent_reward"])

    # Run the RL learning
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
                cont_actions, raw_action = maddpg.getAction(obs_dict)

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
        fitness_mean, fitness_std = maddpg.test(single_env, max_steps=max_steps, loop=20)

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
            print(f"[Episode {ep}] New best model saved! Mean total reward: {best_reward:.2f}")

        # CSV logging
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, mean_total_reward, mean_avg_agent_reward, fitness_mean, fitness_std])

        if ep % plot_every == 0:
            single_env.save_animation(maddpg, f"maddpg_training_{timestamp}")
        print(f"[Episode {ep}] Mean evaluation: {fitness_mean:.2f}, Std evaluation: {fitness_std:.2f}")

        # Exploration decay
        # maddpg.expl_noise = get_expl_noise(ep, num_episodes, start=noise)

    print("Training finished")

    # Close all envs
    for e in envs:
        e.close()
