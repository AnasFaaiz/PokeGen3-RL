import torch 
from stable_baselines3 import DQN
from env import PokemonBattleEnv

OLD_DIM = 34
NEW_DIM = 52

old_model = DQN.load("pokemon_dqn_v3")

env = PokemonBattleEnv()
new_model = DQN(
    "MlpPolicy", env,
    learning_rate=1e-3, buffer_size=5000, learning_starts=50,
    exploration_fraction=0.5, exploration_final_eps=0.1,
    verbose=1, tensorboard_log="./dqn_tensorboard/",
)

old_sd = old_model.policy.q_net.state_dict()
new_sd = new_model.policy.q_net.state_dict()

first_layer_key = "q_net.0.weight"  # confirmed from your inspection output

old_weights = old_sd[first_layer_key]      # shape (64, 34)
new_weights = new_sd[first_layer_key].clone()  # shape (64, 52)
new_weights[:, :OLD_DIM] = old_weights
new_sd[first_layer_key] = new_weights

for key in old_sd:
    if key != first_layer_key:
        new_sd[key] = old_sd[key]

new_model.policy.q_net.load_state_dict(new_sd)
new_model.policy.q_net_target.load_state_dict(new_sd)

new_model.save("pokemon_dqn_v4")
print("Weight transfer complete, saved as pokemon_dqn_v4")
