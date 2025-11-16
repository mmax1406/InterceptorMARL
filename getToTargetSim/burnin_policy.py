import numpy as np
from typing import Dict, Tuple

class ProportionalPolicy:
    """
    Simple proportional control policy for multi-agent navigation.
    Drop-in replacement for MADDPG with getAction() interface.
    """
    
    def __init__(self, env, K: float = 0.1):
        self.env = env
        self.K = K
        self.n_agents = env.n_agents
        self.max_velocity = env.max_velocity
        self.assignment = np.random.permutation(self.n_agents)
        
        # Pre-allocate buffers
        self._actions_array = np.zeros((self.n_agents, 2), dtype=np.float32)
    
    def getAction(self, obs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], None]:
        agent_ids = list(obs.keys())
        
        for i, agent in enumerate(agent_ids):
            agent_obs = obs[agent]
            
            # Extract relative target positions
            # Observation: [rel_agent_pos(2*n), rel_target_pos(2*n), vel(2*n), reached(n)]
            rel_target_start = 2 * self.n_agents
            rel_target_positions = agent_obs[rel_target_start:rel_target_start + 2 * self.n_agents].reshape(self.n_agents, 2)
            
            # Get assigned target's relative position
            target_rel = rel_target_positions[self.assignment[i]]
            
            # Proportional control: action = K * relative_position
            action = self.K * target_rel
            
            # Clip to max velocity
            speed = np.linalg.norm(action)
            if speed > self.max_velocity:
                action = (action / speed) * self.max_velocity
            
            self._actions_array[i] = action
        
        # Convert to dictionary (matching MADDPG format)
        actions_dict = {agent: self._actions_array[i].copy() for i, agent in enumerate(agent_ids)}
        
        return actions_dict, None


# ==================== Example Usage ====================

if __name__ == "__main__":
    import sim_spread
    
    # Create environment
    env = sim_spread.MultiAgentTargetEnv(
        n_agents=3,
        n_targets=3,
        max_steps=200,
        max_velocity=0.5,
        apply_disturbances=False
    )
    
    # Create policy
    policy = ProportionalPolicy(env, K=0.1)
    
    print("Testing ProportionalPolicy")
    print("=" * 50)
    
    # Test single step
    obs, _ = env.reset()
    actions, _ = policy.getAction(obs)
    
    print("Single step test:")
    print(f"  Actions: {list(actions.values())}")
    
    # Test full episode
    print("\nFull episode test:")
    obs, _ = env.reset()
    total_reward = {agent: 0.0 for agent in env.agents}
    
    for step in range(200):
        actions, _ = policy.getAction(obs)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent in env.agents:
            total_reward[agent] += rewards[agent]
        
        if all(terminations.values()) or all(truncations.values()):
            print(f"  Episode finished in {step+1} steps")
            break
    
    print(f"  Total rewards: {[f'{r:.2f}' for r in total_reward.values()]}")
    print(f"  Mean reward: {np.mean(list(total_reward.values())):.2f}")
    
    # Test animation
    print("\nGenerating animation...")
    try:
        animation_path = env.save_animation(policy, "proportional_policy")
        print(f"  ✓ Animation saved to: {animation_path}")
    except Exception as e:
        print(f"  ✗ Animation failed: {e}")
    
    env.close()
    print("\nTest complete!")