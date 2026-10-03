from stable_baselines3 import DQN

new_model = DQN.load("pokemon_dqn_v4")
print(new_model.policy.q_net.state_dict()["q_net.0.weight"].shape)
