import os
import shutil
from train_good import train_good_agent, TRAINED_GOOD_PATH
from train_adversary import train_adversary_agent, TRAINED_ADVERSARY_PATH

# -----------------------------
# CONFIG
# -----------------------------
SELF_PLAY_ITERATIONS = 1      # number of alternating training rounds

# -----------------------------
# Self-play loop
# -----------------------------
def self_play():
    for i in range(1, SELF_PLAY_ITERATIONS + 1):
        print(f"\n==================== SELF-PLAY ROUND {i} ====================")

        # ---------------------
        # Train GOOD agent
        # ---------------------
        print("\n🎯 Training Good Agent...")
        avg_reward_good = train_good_agent(TimeSteps=1_000)
        print(f"✅ Good agent average reward this round: {avg_reward_good:.2f}")

        # ---------------------
        # Train ADVERSARY agent
        # ---------------------
        print("\n👾 Training Adversary Agent...")
        avg_reward_adv = train_adversary_agent(TimeSteps=1_000)
        print(f"✅ Adversary agent average reward this round: {avg_reward_adv:.2f}")

    print("\n==================== SELF-PLAY FINISHED ====================")

if __name__ == "__main__":
    self_play()
