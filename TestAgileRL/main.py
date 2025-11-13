import torch
import numpy as np
import csv
import os

from maddpg import MADDPG  # your class

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Initialize Environment ---
env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
env.reset()

num_agents = len(env.agents)
obs_dims = [env.observation_space(agent).shape[0] for agent in env.agents]
action_dims = [env.action_space(agent).shape[0] for agent in env.agents]

# --- Initialize MADDPG ---
args = dict(
    lr_actor=1e-3,
    lr_critic=1e-3,
    gamma=0.95,
    tau=0.01,
    buffer_size=1000000,
    batch_size=1024,
    device=device,
    update_every=100,
    updates_per_step=1,
)

maddpg = MADDPG(n_agents=num_agents, obs_dims=obs_dims, action_dims=action_dims, args=args)

# --- Logging setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, "training_log.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "total_reward", "avg_agent_reward"])

# --- Training loop ---
num_episodes = 2000
max_steps = 25
print_every = 100

for ep in range(1, num_episodes + 1):
    obs, _ = env.reset()
    maddpg.reset_noise()
    ep_rewards = np.zeros(num_agents)

    for step in range(max_steps):
        obs_list = [obs[agent] for agent in env.agents]
        actions = maddpg.act(obs_list, explore=True)
        actions_dict = {agent: actions[i].astype(np.float32) for i, agent in enumerate(env.agents)}

        next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
        dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
        next_obs_list = [next_obs[agent] for agent in env.agents]
        rewards_list = [rewards.get(agent, 0) for agent in env.agents]
        dones_list = [dones[agent] for agent in env.agents]

        maddpg.step(obs_list, actions, rewards_list, next_obs_list, dones_list)
        obs = next_obs
        ep_rewards += np.array(rewards_list)

        if all(dones_list) or env.agents:
            break

    total_reward = np.sum(ep_rewards)
    avg_agent_reward = np.mean(ep_rewards)

    # --- CSV logging ---
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ep, total_reward, avg_agent_reward])

    if ep % print_every == 0:
        print(f"[Episode {ep}] Total: {total_reward:.2f}, Avg per agent: {avg_agent_reward:.2f}")

print("Training finished ✅")
env.close()
