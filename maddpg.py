from collections import deque, namedtuple
import random
import math
import copy
import time
import argparse
from typing import List, Tuple
import csv
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import imageio

# ------------------------ Utilities ------------------------

def fanin_init(tensor: torch.Tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    else:
        fan_in = np.prod(size[1:])
    bound = 1.0 / math.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def plotPrettyLog(data, window_size=20):
    # Convert list to pandas Series
    series = pd.Series(data)

    # Compute rolling mean and std
    rolling_mean = series.rolling(window_size).mean()
    rolling_std = series.rolling(window_size).std()

    # X-axis: episode indices
    episodes = np.arange(len(data))

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Plot raw data (optional)
    sns.lineplot(x=episodes, y=data, color='lightgray', alpha=0.4, label='Raw Rewards')

    # Plot rolling mean
    sns.lineplot(x=episodes, y=rolling_mean, color='blue', label=f'Mean (window={window_size})')

    # Shaded region for ±1 std
    plt.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     color='blue', alpha=0.2, label='±1 Std. Dev.')

    plt.title("Reward Progress with Sliding Window")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show(block=False)
    plt.savefig('training_reward.png')

def render_static_frames(data, save_path="./Animations/testAlgo.gif"):
    """Render all recorded positions with a fixed camera."""
    frames = []
    cam_range = 1  # Adjust zoom level

    for record in data:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-cam_range, cam_range)
        ax.set_ylim(-cam_range, cam_range)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Plot landmarks
        landmarks = np.array(record["landmarks"])
        if len(landmarks) > 0:
            ax.scatter(landmarks[:, 0], landmarks[:, 1], s=150, c="gray", label="Landmarks")

        # Plot agents
        agents = np.array(record["agents"])
        ax.scatter(agents[:, 0], agents[:, 1], s=150, c="tab:blue", label="Agents")

        # Optional: add step text
        ax.text(-cam_range + 0.05, cam_range - 0.1, f"Step {record['step']}", fontsize=10, color='black')

        # Convert to array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    # Save as GIF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.1)
    print(f"✅ Saved static GIF at: {save_path}")

def rescale_action(action, low, high):
    # action assumed in [-1,1]
    return low + (action + 1.0) * 0.5 * (high - low)

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.size = size
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(*x.shape)
        self.state = x + dx
        return self.state

class RunningMeanStd:
    """Keep running mean and var for normalization (Welford)."""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        x = np.array(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

# ------------------------ Replay Buffer ------------------------

Transition = namedtuple('Transition',
                        ['obs', 'actions', 'rewards', 'next_obs', 'dones'])

class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, seed: int = 0):
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.memory = deque(maxlen=self.buffer_size)
        random.seed(seed)

    def add(self, obs: List[np.ndarray], actions: List[np.ndarray], rewards: List[float], next_obs: List[np.ndarray], dones: List[bool]):
        self.memory.append(Transition(obs, actions, rewards, next_obs, dones))

    def sample(self):
        minibatch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*minibatch))
        return batch

    def __len__(self):
        return len(self.memory)

# ------------------------ Networks ------------------------

class MLPActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes=(32, 32), activation=F.relu, use_layernorm=False):
        super().__init__()
        self.activation = activation
        sizes = [input_dim] + list(hidden_sizes)
        self.layers = nn.ModuleList()
        for i in range(len(sizes)-1):
            lin = nn.Linear(sizes[i], sizes[i+1])
            fanin_init(lin.weight)
            nn.init.constant_(lin.bias, 0.)
            self.layers.append(lin)
            if use_layernorm:
                self.layers.append(nn.LayerNorm(sizes[i+1]))
        # final layer
        self.out = nn.Linear(sizes[-1], action_dim)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.constant_(self.out.bias, 0.)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.activation(x)
            else:
                x = layer(x)
        x = torch.tanh(self.out(x))
        return x

class MLPCritic(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(32, 32), activation=F.relu, use_layernorm=False):
        super().__init__()
        self.activation = activation
        sizes = [input_dim] + list(hidden_sizes)
        self.layers = nn.ModuleList()
        for i in range(len(sizes)-1):
            lin = nn.Linear(sizes[i], sizes[i+1])
            fanin_init(lin.weight)
            nn.init.constant_(lin.bias, 0.)
            self.layers.append(lin)
            if use_layernorm:
                self.layers.append(nn.LayerNorm(sizes[i+1]))
        self.out = nn.Linear(sizes[-1], 1)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.constant_(self.out.bias, 0.)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.activation(x)
            else:
                x = layer(x)
        return self.out(x)

# ------------------------ Agent & MADDPG ------------------------

class Agent:
    def __init__(self, obs_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, weight_decay=0.0,
                 hidden_actor=(256,256), hidden_critic=(512,512), device='cpu', use_layernorm=False):
        self.device = torch.device(device)
        self.actor = MLPActor(obs_dim, action_dim, hidden_actor, use_layernorm=use_layernorm).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic will be created in MADDPG where we know full input dim (global obs+actions)
        self.critic = None
        self.target_critic = None
        self.critic_optimizer = None

        self.action_dim = action_dim

    def act(self, obs: np.ndarray, noise: np.ndarray = None, explore: bool = True):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(x).squeeze(0).cpu().numpy()
        self.actor.train()
        if explore and noise is not None:
            action = action + noise
        action = rescale_action(np.clip(action, -1.0, 1.0), 0, 1.0) #Change to be dynamic
        return action

class MADDPG:
    def __init__(self, n_agents: int, obs_dims: List[int], action_dims: List[int], args: dict = None):
        assert len(obs_dims) == n_agents and len(action_dims) == n_agents
        self.n = n_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.total_obs_dim = sum(obs_dims)
        self.total_action_dim = sum(action_dims)

        # defaults
        default_args = dict(
            buffer_size=1_000_000,
            batch_size=1024,
            gamma=0.99,
            tau=1e-3,
            lr_actor=1e-3,
            lr_critic=1e-3,
            weight_decay=1e-5,
            update_every=100,
            updates_per_step=1,
            critic_hidden=(512,512),
            actor_hidden=(256,256),
            use_layernorm=True,
            device='cpu',
            ou_mu=0.0,
            ou_theta=0.15,
            ou_sigma=0.1,
            clip_grad_norm=0.5,
            reward_scale=1.0,
        )
        if args is not None:
            default_args.update(args)
        self.args = default_args

        self.device = torch.device(self.args['device'])

        # create agents
        self.agents = [Agent(obs_dims[i], action_dims[i], lr_actor=self.args['lr_actor'], lr_critic=self.args['lr_critic'],
                             weight_decay=self.args['weight_decay'], hidden_actor=self.args['actor_hidden'],
                             hidden_critic=self.args['critic_hidden'], device=self.device, use_layernorm=self.args['use_layernorm'])
                       for i in range(self.n)]

        # create centralized critic networks (critic per agent)
        for i, agent in enumerate(self.agents):
            critic_input_dim = self.total_obs_dim + self.total_action_dim
            critic = MLPCritic(critic_input_dim, hidden_sizes=self.args['critic_hidden'], use_layernorm=self.args['use_layernorm']).to(self.device)
            target_critic = copy.deepcopy(critic).to(self.device)
            agent.critic = critic
            agent.target_critic = target_critic
            agent.critic_optimizer = optim.Adam(critic.parameters(), lr=self.args['lr_critic'], weight_decay=self.args['weight_decay'])

        # replay buffer shared
        self.buffer = ReplayBuffer(self.args['buffer_size'], self.args['batch_size'])

        # OU noise for each agent
        self.noises = [OUNoise((action_dims[i],), mu=self.args['ou_mu'], theta=self.args['ou_theta'], sigma=self.args['ou_sigma']) for i in range(self.n)]

        # normalization
        self.obs_rms = [RunningMeanStd(shape=(d,)) for d in obs_dims]
        self.ret_rms = RunningMeanStd(shape=())

        # bookkeeping
        self.total_steps = 0
        self.updates_done = 0

    def reset_noise(self):
        for n in self.noises:
            n.reset()

    def scale_obs(self, obs_list: List[np.ndarray]):
        out = []
        for i, obs in enumerate(obs_list):
            mean = self.obs_rms[i].mean
            std = np.sqrt(self.obs_rms[i].var) + 1e-8
            out.append((obs - mean) / std)
        return out

    def act(self, obs_list: List[np.ndarray], explore: bool = True):
        assert len(obs_list) == self.n
        obs_norm = self.scale_obs(obs_list)
        actions = []
        for i, agent in enumerate(self.agents):
            noise = None
            if explore:
                noise = self.noises[i].sample()
            a = agent.act(obs_norm[i], noise=noise, explore=explore)
            actions.append(a)
        return actions

    def step(self, obs, actions, rewards, next_obs, dones):
        self.total_steps += 1
        # update running stats
        for i in range(self.n):
            self.obs_rms[i].update(np.array([obs[i]]))
        self.ret_rms.update(np.array([sum(rewards)]))

        self.buffer.add(obs, actions, rewards, next_obs, dones)

        if len(self.buffer) >= self.args['batch_size'] and self.total_steps % self.args['update_every'] == 0:
            for _ in range(self.args['updates_per_step']):
                self.update()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update(self):
        batch = self.buffer.sample()
        batch_size = len(batch.obs)

        obs_batch = [[] for _ in range(self.n)]
        next_obs_batch = [[] for _ in range(self.n)]
        actions_batch = [[] for _ in range(self.n)]
        rewards_batch = [[] for _ in range(self.n)]
        dones_batch = [[] for _ in range(self.n)]

        for b in range(batch_size):
            obs_b = batch.obs[b]
            next_obs_b = batch.next_obs[b]
            actions_b = batch.actions[b]
            rewards_b = batch.rewards[b]
            dones_b = batch.dones[b]
            for i in range(self.n):
                obs_batch[i].append(obs_b[i])
                next_obs_batch[i].append(next_obs_b[i])
                actions_batch[i].append(actions_b[i])
                rewards_batch[i].append(rewards_b[i])
                dones_batch[i].append(float(dones_b[i]))

        obs_tensor = [torch.tensor(np.array(obs_batch[i]), dtype=torch.float32, device=self.device) for i in range(self.n)]
        next_obs_tensor = [torch.tensor(np.array(next_obs_batch[i]), dtype=torch.float32, device=self.device) for i in range(self.n)]
        actions_tensor = [torch.tensor(np.array(actions_batch[i]), dtype=torch.float32, device=self.device) for i in range(self.n)]
        rewards_tensor = [torch.tensor(np.array(rewards_batch[i]), dtype=torch.float32, device=self.device).unsqueeze(1) for i in range(self.n)]
        dones_tensor = [torch.tensor(np.array(dones_batch[i]), dtype=torch.float32, device=self.device).unsqueeze(1) for i in range(self.n)]

        for i in range(self.n):
            mean = torch.tensor(self.obs_rms[i].mean, dtype=torch.float32, device=self.device)
            std = torch.tensor(np.sqrt(self.obs_rms[i].var) + 1e-8, dtype=torch.float32, device=self.device)
            obs_tensor[i] = (obs_tensor[i] - mean) / std
            next_obs_tensor[i] = (next_obs_tensor[i] - mean) / std

        obs_cat = torch.cat(obs_tensor, dim=1)
        next_obs_cat = torch.cat(next_obs_tensor, dim=1)
        actions_cat = torch.cat(actions_tensor, dim=1)

        with torch.no_grad():
            next_actions = []
            for i in range(self.n):
                a = self.agents[i].target_actor(next_obs_tensor[i])
                next_actions.append(a)
            next_actions_cat = torch.cat(next_actions, dim=1)

        for i in range(self.n):
            agent = self.agents[i]
            agent.critic_optimizer.zero_grad()
            with torch.no_grad():
                q_next = agent.target_critic(torch.cat([next_obs_cat, next_actions_cat], dim=1))
                q_target = rewards_tensor[i] * self.args['reward_scale'] + (1.0 - dones_tensor[i]) * (self.args['gamma'] * q_next)
            q_expected = agent.critic(torch.cat([obs_cat, actions_cat], dim=1))
            critic_loss = F.mse_loss(q_expected, q_target)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.args['clip_grad_norm'])
            agent.critic_optimizer.step()

            agent.actor_optimizer.zero_grad()
            curr_actions = []
            for j in range(self.n):
                if j == i:
                    curr_actions.append(agent.actor(obs_tensor[j]))
                else:
                    with torch.no_grad():
                        curr_actions.append(self.agents[j].actor(obs_tensor[j]))
            curr_actions_cat = torch.cat(curr_actions, dim=1)
            actor_loss = -agent.critic(torch.cat([obs_cat, curr_actions_cat], dim=1)).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.args['clip_grad_norm'])
            agent.actor_optimizer.step()

            self.soft_update(agent.actor, agent.target_actor, self.args['tau'])
            self.soft_update(agent.critic, agent.target_critic, self.args['tau'])

        self.updates_done += 1

    def save(self, path_prefix: str):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f"{path_prefix}//agent{i}_actor.pth")
            torch.save(agent.critic.state_dict(), f"{path_prefix}//agent{i}_critic.pth")

    def load(self, path_prefix: str):
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f"{path_prefix}//agent{i}_actor.pth", map_location=self.device))
            agent.critic.load_state_dict(torch.load(f"{path_prefix}//agent{i}_critic.pth", map_location=self.device))
            agent.target_actor = copy.deepcopy(agent.actor)
            agent.target_critic = copy.deepcopy(agent.critic)

# ------------------------ CLI and runnable example ------------------------

if __name__ == '__main__':
# ------------------------Im saving it to remember how to use args ------------------------
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train', action='store_true', help='Run training example')
    # parser.add_argument('--gpu', action='store_true', help='Force GPU if available')
    # parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
    # parser.add_argument('--save-prefix', type=str, default='maddpg_ckpt', help='Checkpoint prefix')
    # args = parser.parse_args()

    try:
        from pettingzoo.mpe import simple_spread_v3  # multi-agent continuous env
        print("pettingzoo imported successfully.")
    except ImportError:
        raise ImportError("The 'some_library' is not installed. Please install it using 'pip install some_library'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, plot = False, True

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
    max_steps = 50
    save_dir = "checkpoints"

    if train:
        # --- Logging setup ---
        log_dir = "logs"
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
            reward_list.append(total_reward)

            # --- Save best model ---
            if total_reward > best_reward:
                os.makedirs(save_dir, exist_ok=True)
                best_reward = total_reward
                maddpg.save(save_dir)
                print(f"[Episode {ep}] 🏆 New best model saved! Total reward: {best_reward:.2f}")

            # --- CSV logging ---
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep, total_reward, avg_agent_reward])

            if ep % print_every == 0:
                print(f"[Episode {ep}] Total: {total_reward:.2f}, Avg per agent: {avg_agent_reward:.2f}")

        plotPrettyLog(reward_list)
        print("Training finished ✅")

    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=max_steps, continuous_actions=True, render_mode="rgb_array")

    frames = []
    if plot:
        maddpg.load(save_dir)
        with torch.no_grad():
            obs, _ = env.reset()
            maddpg.reset_noise()
            ep_rewards = np.zeros(num_agents)
            data, agent_positions, landmark_positions = [], [], []

            for step in range(max_steps):
                obs_list = [obs[agent] for agent in env.agents]
                actions = maddpg.act(obs_list)
                actions_dict = {agent: actions[i].astype(np.float32) for i, agent in enumerate(env.agents)}
                next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
                dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

                agent_positions = [agent.state.p_pos.copy() for agent in env.unwrapped.world.agents]
                landmark_positions = [lm.state.p_pos.copy() for lm in env.unwrapped.world.landmarks]
                data.append({
                    "step": step,
                    "agents": agent_positions,
                    "landmarks": landmark_positions
                })

                # frame = env.render()
                # frames.append(frame)
                if len(env.agents) == 0 or all(dones.values()):
                    break

        # Save the gif to specified path
        gif_path = "./Animations/"
        os.makedirs(gif_path, exist_ok=True)
        render_static_frames(data)
        # imageio.mimwrite(os.path.join(gif_path, "testAlgo.gif"), frames, duration=10)


    env.close()
