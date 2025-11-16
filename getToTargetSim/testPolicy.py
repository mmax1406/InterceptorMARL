import os
import glob
import numpy as np
import torch
from agilerl.algorithms.maddpg import MADDPG
import sim_spread
import burnin_policy

# ------------------------ Configuration ------------------------

# Set to True to automatically find the latest checkpoint
AUTO_FIND_LATEST = True

# Test configuration
MAX_STEPS = 200
NUM_AGENTS = 3
SAVE_ANIMATION = True
ANIMATION_FILENAME = "maddpg_test"  # Will be appended with timestamp

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------ Helper Functions ------------------------

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the most recent checkpoint file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, checkpoint_dir)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {full_path}")
    
    # Find all .pt files
    checkpoint_files = glob.glob(os.path.join(full_path, "*.pt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {full_path}")
    
    # Get the most recent file by modification time
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

def safe_reset(env):
    """Reset environment and return observation dict."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out

def load_maddpg_agent(checkpoint_path, env, device):
    """Load MADDPG agent from checkpoint."""
    agent_ids = env.agents[:]
    
    # Get state and action dimensions
    state_dims = [env.observation_space(agent).shape for agent in agent_ids]
    action_dims = [env.action_space(agent).shape[0] for agent in agent_ids]
    
    min_action = env.action_space(agent_ids[0]).low[0]
    max_action = env.action_space(agent_ids[0]).high[0]
    
    # Initialize MADDPG agent with same configuration as training
    maddpg = MADDPG(
        state_dims=state_dims,
        action_dims=action_dims,
        one_hot=False,
        n_agents=len(agent_ids),
        agent_ids=agent_ids,
        max_action=[[max_action] * action_dims[i] for i in range(len(agent_ids))],
        min_action=[[min_action] * action_dims[i] for i in range(len(agent_ids))],
        discrete_actions=False,
        net_config={
            'arch': 'mlp',
            'h_size': [256, 256]
        },
        batch_size=512,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.97,
        tau=0.017,
        expl_noise=0.0,  # No exploration during testing
        device=device,
    )
    
    # Load checkpoint
    maddpg.loadCheckpoint(checkpoint_path)
    print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    
    return maddpg

# ------------------------ Main Script ------------------------

if __name__ == '__main__':
    
    print("=" * 70)
    print("MADDPG Agent Testing Script")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    # Find checkpoint
    if AUTO_FIND_LATEST:
        try:
            checkpoint_path = find_latest_checkpoint()
            print(f"Auto-detected latest checkpoint:")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please set AUTO_FIND_LATEST=False and specify MANUAL_CHECKPOINT_PATH")
            exit(1)
    
    # Create environment
    env = sim_spread.MultiAgentTargetEnv(
        n_agents=NUM_AGENTS,
        n_targets=NUM_AGENTS,
        max_steps=MAX_STEPS,
        max_velocity=0.5,
        apply_disturbances=False
    )
      
    # Load agent
    policy = load_maddpg_agent(checkpoint_path, env, DEVICE)
    # policy = burnin_policy.ProportionalPolicy(env, K=0.1)

    animation_path = env.save_animation(policy, filename=ANIMATION_FILENAME)
    
    print('Animation created')
    # Close environment
    env.close()