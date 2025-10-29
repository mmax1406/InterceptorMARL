from intercept_env import env

environment = env(N_adversaries=3, M_good=5, width_ratio=3.0)
observations, info = environment.reset()

for step in range(200):
    actions = {agent: environment.action_spaces[agent].sample() for agent in environment.agents}
    obs, rewards, term, trunc, infos = environment.step(actions)
    environment.render()
    if not environment.agents:
        print("Game over")
        break
