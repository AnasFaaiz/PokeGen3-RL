import torch
from stable_baselines3 import DQN
from env import PokemonBattleEnv

OLD_ACTIONS = 4
NEW_ACTIONS = 10

old_model = DQN.load("pokemon_dqn_v5")

env = PokemonBattleEnv()  # action_space must already be Discrete(10) here
new_model = DQN(
    "MlpPolicy", env,
    learning_rate=1e-3, buffer_size=5000, learning_starts=50,
    exploration_fraction=0.5, exploration_final_eps=0.1,
    verbose=1, tensorboard_log="./dqn_tensorboard/",
)

old_sd = old_model.policy.q_net.state_dict()
new_sd = new_model.policy.q_net.state_dict()

first_layer_key = "q_net.0.weight"
output_weight_key = "q_net.4.weight"
output_bias_key = "q_net.4.bias"

# input layer: 64 -> 64, unchanged, direct copy
new_sd[first_layer_key] = old_sd[first_layer_key]

# output layer: 4 -> 10 actions, copy existing 4, leave 6 new ones random-init
old_out_w = old_sd[output_weight_key]
new_out_w = new_sd[output_weight_key].clone()
new_out_w[:OLD_ACTIONS, :] = old_out_w
new_sd[output_weight_key] = new_out_w

old_out_b = old_sd[output_bias_key]
new_out_b = new_sd[output_bias_key].clone()
new_out_b[:OLD_ACTIONS] = old_out_b
new_sd[output_bias_key] = new_out_b

for key in old_sd:
    if key not in (first_layer_key, output_weight_key, output_bias_key):
        new_sd[key] = old_sd[key]

new_model.policy.q_net.load_state_dict(new_sd)
new_model.policy.q_net_target.load_state_dict(new_sd)

new_model.save("pokemon_dqn_v6")
print("Transfer complete: 64-dim input unchanged, output expanded 4->10, saved as pokemon_dqn_v6")
