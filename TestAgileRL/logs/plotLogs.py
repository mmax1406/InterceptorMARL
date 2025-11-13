import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plotPrettyLogHavingMeanAndStd(csv_path, save_path=None):
    # Read CSV
    df = pd.read_csv(csv_path)

    if 'episode' not in df.columns or 'total_reward' not in df.columns:
        raise ValueError("CSV must contain 'episode' and 'total_reward' columns.")

    episodes = df['episode']
    rewards = df['total_reward']
    mean = df['']
    std = df['']

    # Set up figure
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Plot data
    sns.lineplot(x=episodes, y=mean, label=f'Mean')
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.2, label='±1 Std. Dev.')

    plt.ylim([-150, 0])
    plt.title("Reward Progress with Sliding Window")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    # Determine save path
    if save_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_dir = os.path.dirname(csv_path) or '.'
        save_path = os.path.join(save_dir, f"{base_name}_plot.png")

    plt.savefig(save_path)
    plt.show(block=False)

    print(f"✅ Plot saved to: {save_path}")

def plotPrettyLog(csv_path, window_size=100, save_path=None):
    # Read CSV
    df = pd.read_csv(csv_path)

    if 'episode' not in df.columns or 'total_reward' not in df.columns:
        raise ValueError("CSV must contain 'episode' and 'total_reward' columns.")

    episodes = df['episode']
    rewards = df['total_reward']

    # Compute rolling statistics
    rolling_mean = rewards.rolling(window_size).mean()
    rolling_std = rewards.rolling(window_size).std()

    # Set up figure
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Plot data
    sns.lineplot(x=episodes, y=rewards, alpha=0.35, label='Raw Rewards')
    sns.lineplot(x=episodes, y=rolling_mean, label=f'Mean (window={window_size})')
    plt.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     alpha=0.2, label='±1 Std. Dev.')

    plt.ylim([-150, 0])
    plt.title("Reward Progress with Sliding Window")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    # Determine save path
    if save_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_dir = os.path.dirname(csv_path) or '.'
        save_path = os.path.join(save_dir, f"{base_name}_plot.png")

    plt.savefig(save_path)
    plt.show(block=False)

    print(f"✅ Plot saved to: {save_path}")


if __name__ == "__main__":
    # Example usage:
    plotPrettyLog("training_log.csv", window_size=100)
    plotPrettyLogHavingMeanAndStd("training_log_agileRL.csv")
    plotPrettyLogHavingMeanAndStd("training_log_parallel.csv")
