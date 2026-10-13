from stable_baselines3 import DQN
m = DQN.load("pokemon_dqn_v5")
print(m.policy.q_net.state_dict()["q_net.0.weight"].shape)
