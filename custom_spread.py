import numpy as np
import os 
from datetime import datetime
import torch
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.mpe import simple_tag_v3

class CustomSpreadEnv:
    def __init__(self, map_size=1.0, *args, **kwargs):
        # Create the underlying parallel env
        self.env = simple_spread_v3.parallel_env(*args, **kwargs)
        self.map_size = map_size
        self.eps = 0.1

    # ---------- PARALLEL CONSTRUCTOR ----------
    @classmethod
    def parallel_env(cls, *args, **kwargs):
        return cls(*args, **kwargs)


    # ---------- RESET ----------
    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)

        world = self.env.unwrapped.world

        # --- Modify map bounds ---
        if hasattr(world, "bounds"):
            world.bounds = [-self.map_size, self.map_size]

        # --- Spawn agents on the left ---
        n_agents = len(world.agents)
        left_x = -self.map_size + self.eps

        # Spread Y positions uniformly across map height
        ys = np.linspace(-self.map_size * 0.8, self.map_size * 0.8, n_agents)

        for agent, y in zip(world.agents, ys):
            agent.state.p_pos = np.array([left_x, y], dtype=np.float32)
            agent.state.p_vel = np.zeros(2, dtype=np.float32)

        # --- Spawn targets/landmarks on the right ---
        for landmark in world.landmarks:
            x = np.random.uniform(0.0 + self.eps, self.map_size)
            y = np.random.uniform(-self.map_size, self.map_size)
            landmark.state.p_pos = np.array([x, y], dtype=np.float32)
            landmark.state.p_vel = np.zeros(2, dtype=np.float32)

        return obs

    # ---------- STEP ----------
    def step(self, action_dict):
        return self.env.step(action_dict)

    # ---------- ATTRIBUTE FORWARDING ----------
    def __getattr__(self, name):
        env = object.__getattribute__(self, "env")
        return getattr(env, name)
    
    # ---------- ANIMATION ----------
    def save_animation(self, policy, filename=None):
        import matplotlib.pyplot as plt
        import imageio
        import matplotlib as mpl

        script_dir = os.path.dirname(os.path.abspath(__file__))
        animations_dir = os.path.join(script_dir, 'Animations')
        os.makedirs(animations_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if filename is None:
            filename = f'multi_agent_env_{timestamp}.gif'
        else:
            filename = f'{filename}_multi_agent_env.gif'

        save_path = os.path.join(animations_dir, filename)

        obs, _ = self.reset()
        frames = []

        world = self.env.unwrapped.world
        agents = world.agents
        landmarks = world.landmarks

        # Use environment horizon if available, otherwise fallback
        max_steps = getattr(self.env, "max_cycles", 200)

        # Styling
        mpl.rcParams.update({
            "axes.facecolor": "#f8f9fa",
            "figure.facecolor": "#f8f9fa",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.edgecolor": "#cccccc",
            "font.family": "DejaVu Sans",
            "font.size": 11,
        })

        for step in range(max_steps):
            # ------ Extract positions & velocities
            agent_positions = np.array([a.state.p_pos for a in agents])
            agent_vels = np.array([a.state.p_vel for a in agents])

            landmark_positions = np.array([l.state.p_pos for l in landmarks])

            # -------- Create plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(-self.map_size, self.map_size)
            ax.set_ylim(-self.map_size, self.map_size)
            ax.set_aspect("equal")
            ax.set_title(f"Simulation Step {step}", fontsize=13, fontweight="bold", color="#333")

            # Landmarks
            if len(landmark_positions):
                ax.scatter(
                    landmark_positions[:, 0], landmark_positions[:, 1],
                    s=80, c="tab:red", edgecolors="black", linewidths=1.2, zorder=2
                )

            # Agents
            ax.scatter(
                agent_positions[:, 0], agent_positions[:, 1],
                s=200, c="tab:blue", edgecolors="white", linewidths=1.5, zorder=3
            )

            # Velocities (arrows)
            for pos, vel in zip(agent_positions, agent_vels):
                if np.linalg.norm(vel) > 1e-6:
                    ax.arrow(pos[0], pos[1], vel[0] * 0.3, vel[1] * 0.3,
                            head_width=0.08, head_length=0.1,
                            alpha=0.7, color="black", zorder=4)

            # Render frame
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(frame)
            plt.close(fig)

            # ------- Step the agents
            with torch.no_grad():
                cont_actions, _ = policy.getAction(obs)
                actions_dict = {agent_name: cont_actions[agent_name] for agent_name in self.agents}

            obs, rewards, terminations, truncations, infos = self.step(actions_dict)

            if any(terminations.values()) or any(truncations.values()):
                # Hold last frame
                frames.extend([frames[-1]] * 10)
                break

        # Save GIF
        imageio.mimsave(save_path, frames, duration=0.07)
        return save_path


if __name__ == "__main__":
    my_map_size = 2.0
    custom_env = CustomSpreadEnv(
        map_size=my_map_size,
        render_mode="human",
        continuous_actions=True
    )

    custom_env.reset()

    for _ in range(25):
        actions = {
            agent: custom_env.action_space(agent).sample()
            for agent in custom_env.agents
        }
        obs, rewards, terminations, truncations, infos = custom_env.step(actions)

        custom_env.render()





