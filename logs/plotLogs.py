import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotPrettyLog(data, window_size=100, save_path='logs//training_reward_agileRL.png'):
    """Plot training rewards with rolling mean and std."""
    series = pd.Series(data)
    rolling_mean = series.rolling(window_size).mean()
    rolling_std = series.rolling(window_size).std()
    episodes = np.arange(len(data))

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(x=episodes, y=data, alpha=0.35, label='Raw Rewards')
    sns.lineplot(x=episodes, y=rolling_mean, label=f'Mean (window={window_size})')
    plt.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     alpha=0.2, label='±1 Std. Dev.')
    plt.title("Reward Progress with Sliding Window")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show(block=False)


if __name__ == "__main__":
    plotPrettyLog(data, window_size=100, save_path='logs//training_reward_agileRL.png')