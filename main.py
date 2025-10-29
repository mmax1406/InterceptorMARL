import torch
import numpy as np
import os
from sharedPolicyWrapper import SharedPolicyWrapper
from intercept_env import env
from helperScripts import ActorCritic as Policy

# ---------------- CONFIG ----------------
GOOD_MODEL_PATH = "good_policy.pt"
ADV_MODEL_PATH = "adversary_policy.pt"
N_ADVERSARIES = 1
M_GOOD = 1
WIDTH_RATIO = 5.0
EPISODE_STEPS = 300
RENDER = True

# ---------------- ENV SETUP ----------------
base_env = env(N_adversaries=N_ADVERSARIES, M_good=M_GOOD, width_ratio=WIDTH_RATIO)
wrapped_env = SharedPolicyWrapper(base_env)

# Reset environment
obs, info = wrapped_env.reset()
print("Initial observation shape:", obs.shape)

# ---------------- LOAD POLICIES ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs_dim = wrapped_env.observation_space.shape[0]
act_dim = wrapped_env.action_space.shape[0]

# Try to load good policy
use_good_policy = False
if os.path.exists(GOOD_MODEL_PATH):
    try:
        good_policy = Policy(obs_dim, act_dim).to(device)
        good_policy.load_state_dict(torch.load(GOOD_MODEL_PATH, map_location=device))
        good_policy.eval()
        use_good_policy = True
        print(f"✅ Loaded good policy: {GOOD_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load good policy: {e}")
        print("   Using random actions for good agents")
else:
    print(f"⚠️ Good policy not found: {GOOD_MODEL_PATH}")
    print("   Using random actions for good agents")

# Try to load adversary policy
use_adv_policy = False
if os.path.exists(ADV_MODEL_PATH):
    try:
        adv_policy = Policy(obs_dim, act_dim).to(device)
        adv_policy.load_state_dict(torch.load(ADV_MODEL_PATH, map_location=device))
        adv_policy.eval()
        use_adv_policy = True
        print(f"✅ Loaded adversary policy: {ADV_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load adversary policy: {e}")
        print("   Using random actions for adversary agents")
else:
    print(f"⚠️ Adversary policy not found: {ADV_MODEL_PATH}")
    print("   Using random actions for adversary agents")

# ---------------- TEST LOOP ----------------
episode_reward_good = 0
episode_reward_adv = 0

for step in range(EPISODE_STEPS):
    # Convert observations to tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    # Split observations into groups
    good_indices = [i for i, a in enumerate(wrapped_env.agents) if "good" in a]
    adv_indices = [i for i, a in enumerate(wrapped_env.agents) if "adversary" in a]

    # Get good actions
    if use_good_policy and len(good_indices) > 0:
        good_obs = obs_tensor[good_indices]
        with torch.no_grad():
            good_actions, _ = good_policy.act(good_obs)
        good_actions = good_actions.cpu().numpy()
    else:
        # Random actions for good agents
        good_actions = np.random.uniform(-1, 1, size=(len(good_indices), act_dim))

    # Get adversary actions
    if use_adv_policy and len(adv_indices) > 0:
        adv_obs = obs_tensor[adv_indices]
        with torch.no_grad():
            adv_actions, _ = adv_policy.act(adv_obs)
        adv_actions = adv_actions.cpu().numpy()
    else:
        # Random actions for adversary agents
        adv_actions = np.random.uniform(-1, 1, size=(len(adv_indices), act_dim))

    # Combine actions in correct agent order
    actions = []
    good_i, adv_i = 0, 0
    for a in wrapped_env.agents:
        if "good" in a:
            actions.append(good_actions[good_i])
            good_i += 1
        else:
            actions.append(adv_actions[adv_i])
            adv_i += 1
    actions = np.stack(actions) if actions else np.array([])

    # Step environment
    next_obs, reward, done, infos = wrapped_env.step(actions)

    # Compute separate team rewards
    for i, a in enumerate(wrapped_env.agents):
        if "good" in a:
            episode_reward_good += reward[i]
        else:
            episode_reward_adv += reward[i]

    if RENDER:
        base_env.render()

    if done:
        break

    obs = next_obs

print("\n✅ Testing complete")

# Summary
print("\n" + "="*50)
print("POLICY USAGE SUMMARY")
print("="*50)
print(f"Good agents: {'Trained Policy' if use_good_policy else 'Random Actions'}")
print(f"Adversary agents: {'Trained Policy' if use_adv_policy else 'Random Actions'}")
print("="*50)
print(f"\n{'='*50}")
print(f"Episode finished after {step+1} steps")
print(f"{'='*50}")
print(f"Good team total reward: {episode_reward_good:.2f}")
print(f"Adversary team total reward: {episode_reward_adv:.2f}")
print(f"{'='*50}")