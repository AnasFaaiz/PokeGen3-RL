import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO


# 1. Dummy environment matching your exact V7 shapes (no emulator/socket connection required)
class DummyPokemonEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(285,), dtype=np.float32
        )

    def action_masks(self):
        return np.ones(17, dtype=bool)

    def reset(self, seed=None, options=None):
        return np.zeros(285, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(285, dtype=np.float32), 0.0, True, False, {}


def main():
    old_model_path = "pokemon_dqn_v6.zip" if os.path.exists("pokemon_dqn_v6.zip") else "pokemon_dqn_v6"
    print(f"Loading existing model from {old_model_path}...")
    
    old_model = DQN.load(old_model_path)
    old_sd = old_model.policy.q_net.state_dict()

    env = DummyPokemonEnv()
    new_model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        ent_coef=0.05,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
    )
    new_sd = new_model.policy.state_dict()

    print("\n--- Transferring Input Layer ---")
    old_in_w = old_sd["q_net.0.weight"]
    old_in_b = old_sd["q_net.0.bias"]
    old_in_features = old_in_w.shape[1]
    print(f"Detected old observation input dim: {old_in_features} -> New observation dim: 285")

    # Map input weights to Policy (Actor) and Value (Critic) nets; remaining features stay 0
    new_in_w = new_sd["mlp_extractor.policy_net.0.weight"].clone()
    new_in_w[:, :old_in_features] = old_in_w
    new_sd["mlp_extractor.policy_net.0.weight"] = new_in_w
    new_sd["mlp_extractor.value_net.0.weight"] = new_in_w.clone()

    new_sd["mlp_extractor.policy_net.0.bias"] = old_in_b
    new_sd["mlp_extractor.value_net.0.bias"] = old_in_b.clone()

    print("\n--- Transferring Hidden Layer ---")
    new_sd["mlp_extractor.policy_net.2.weight"] = old_sd["q_net.2.weight"]
    new_sd["mlp_extractor.value_net.2.weight"] = old_sd["q_net.2.weight"]
    new_sd["mlp_extractor.policy_net.2.bias"] = old_sd["q_net.2.bias"]
    new_sd["mlp_extractor.value_net.2.bias"] = old_sd["q_net.2.bias"]

    print("\n--- Transferring Output Layer (Actions) ---")
    old_out_w = old_sd["q_net.4.weight"]
    old_out_b = old_sd["q_net.4.bias"]
    num_old_actions = old_out_w.shape[0]
    print(f"Detected old action count: {num_old_actions} -> New action count: 17")

    # Dynamically copy detected actions (0 to num_old_actions)
    new_out_w = new_sd["action_net.weight"].clone()
    new_out_w[:num_old_actions, :] = old_out_w
    new_sd["action_net.weight"] = new_out_w

    new_out_b = new_sd["action_net.bias"].clone()
    new_out_b[:num_old_actions] = old_out_b
    new_sd["action_net.bias"] = new_out_b

    print("\nLoading weights into MaskablePPO policy...")
    new_model.policy.load_state_dict(new_sd)

    output_model_name = "pokemon_ppo_masked_v7"
    new_model.save(output_model_name)
    print(f"\nTransfer complete! Saved weights as {output_model_name}.zip")


if __name__ == "__main__":
    main()
