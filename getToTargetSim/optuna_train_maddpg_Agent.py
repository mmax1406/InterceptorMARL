import os
import csv
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
import sim_spread
import optuna
from optuna.pruners import MedianPruner
from tqdm import tqdm

# ------------------------ Helpers ------------------------

def safe_reset(env):
    """Reset a PettingZoo parallel env and return the obs dict (compat with different returns)."""
    out = env.reset()
    # Some versions return (obs, infos) when render_mode is set; support either.
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out

def get_expl_noise(episode, total_episodes, start=0.5, end=0.05, decay_fraction=0.8):
    decay_episodes = int(total_episodes * decay_fraction)
    if episode >= decay_episodes:
        return end
    else:
        # Linear decay
        return start - (start - end) * (episode / decay_episodes)

# ------------------------ Main Script ------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Multi-env (vectorized) settings ---
NUM_ENVS = 4                     
max_steps = 200
num_agents = 3                   
envs = []
for i in range(NUM_ENVS):
    e = sim_spread.MultiAgentTargetEnv( n_agents=num_agents, n_targets=num_agents, max_velocity=0.5,apply_disturbances=False, action_repeat=5)
    envs.append(e)

single_env = sim_spread.MultiAgentTargetEnv( n_agents=num_agents, n_targets=num_agents, max_velocity=0.5,apply_disturbances=False, action_repeat=5)

# Reset all envs and check agent list
obs_list = [safe_reset(e) for e in envs]
agent_ids = envs[0].agents[:]   # order should be same for all envs

# Get state and action dims from first env agent
state_dims = [envs[0].observation_space(agent).shape for agent in agent_ids]
action_dims = [envs[0].action_space(agent).shape[0] for agent in agent_ids]

discrete_actions = False
min_action = envs[0].action_space(agent_ids[0]).low[0]
max_action = envs[0].action_space(agent_ids[0]).high[0]

# Network Params
num_episodes = 2_000
print_every = 100
best_reward = -999999
update_every = 100    # perform learning once every X global steps (across envs)
total_steps = 0

def objective(trial):
    # --- Sample hyperparameters ---
    lr_actor = trial.suggest_loguniform("lr_actor", 1e-4, 1e-2)
    lr_critic = trial.suggest_loguniform("lr_critic", 1e-4, 1e-2)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.99)
    tau = trial.suggest_uniform("tau", 0.005, 0.02)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 128, 256])
    expl_start = trial.suggest_uniform("expl_start", 0.3, 0.7)
    expl_end = trial.suggest_uniform("expl_end", 0.01, 0.1)

    # --- Initialize MADDPG ---
    maddpg = MADDPG(
        state_dims=state_dims,
        action_dims=action_dims,
        one_hot=False,
        n_agents=num_agents,
        agent_ids=agent_ids,
        max_action=[[max_action] * action_dims[i] for i in range(num_agents)],
        min_action=[[min_action] * action_dims[i] for i in range(num_agents)],
        discrete_actions=discrete_actions,
        net_config={'arch': 'mlp', 'h_size': [hidden_size, hidden_size]},
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        expl_noise=expl_start,
        device=device,
    )

    memory = MultiAgentReplayBuffer(
        memory_size=1_000_000,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agent_ids,
        device=device,
    )

    num_episodes_trial = 1_000
    max_steps_trial = max_steps
    total_steps = 0
    best_reward_trial = -999999

    for ep in tqdm(range(1, num_episodes_trial + 1)):
        obs_list = [safe_reset(e) for e in envs]
        ep_rewards_per_env = np.zeros((NUM_ENVS, num_agents))

        for step in range(max_steps_trial):
            total_steps += 1
            for env_idx, env in enumerate(envs):
                obs = obs_list[env_idx]
                obs_dict = {agent_id: obs[agent_id] for agent_id in agent_ids}
                cont_actions, _ = maddpg.getAction(obs_dict)
                actions_dict = {agent_id: cont_actions[agent_id] for agent_id in agent_ids}
                next_obs, rewards, terminations, truncations, _ = env.step(actions_dict)
                dones = {agent: terminations[agent] or truncations[agent] for agent in agent_ids}

                memory.save2memory(obs, cont_actions, rewards, next_obs, dones)
                ep_rewards_per_env[env_idx] += np.array([rewards[agent] for agent in agent_ids])
                obs_list[env_idx] = next_obs

                if all(dones.values()) or len(env.agents) == 0:
                    obs_list[env_idx] = safe_reset(env)

            if len(memory) >= maddpg.batch_size and total_steps % update_every == 0:
                experiences = memory.sample(maddpg.batch_size)
                maddpg.learn(experiences)

        # --- Evaluate MADDPG after each episode ---
        fitness_mean, _ = maddpg.test(single_env, max_steps=max_steps_trial, loop=5)

        # Exploration decay
        maddpg.expl_noise = get_expl_noise(ep, num_episodes_trial, start=expl_start, end=expl_end)

        # Report to Optuna
        trial.report(fitness_mean, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if fitness_mean > best_reward_trial:
            best_reward_trial = fitness_mean

    return best_reward_trial


if __name__ == "__main__":
    # Use median pruner to prune underperforming trials
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=10, interval_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=20)  # increase n_trials if you want more search

    # Export all trial info to CSV
    trials_data = []
    for t in study.trials:
        trial_info = {
            "trial_number": t.number,
            "value": t.value,
            "state": t.state
        }
        trial_info.update(t.params)
        trials_data.append(trial_info)

    df = pd.DataFrame(trials_data)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "optuna_trials_results.csv")
    df.to_csv(save_dir, index=False)
    print("Trial info saved to optuna_trials_results.csv")

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
