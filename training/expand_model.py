"""
One-time script: expand an existing MaskablePPO checkpoint's input layer
from 285 to 417 dims, preserving every already-learned weight.

Why this is safe: env.py's new bench-effectiveness features were APPENDED
after the original 188 (active battlers) + 1 (weather) = 189 features,
not interleaved into them. So old observation index i still means exactly
what it meant before, for all i < 189 (the 189th value, weather, also kept
its old spot). Only new input connections (for the new 228 bench values,
indices 189-416) need to be added -- every other weight in the network
(all hidden layers, policy head, value head) is left completely untouched.

Run this ONCE, after confirming env.py's new 417-dim observation_space is
in place, before starting your next training run:

    python expand_model.py

Produces pokemon_ppo_masked_v7_expanded.zip -- point train_dqn.py's
model_name at that file (or rename it) to continue training from here.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Paths are relative to this script's own location, not whatever directory
# you happen to run `python ...` from, so this works whether you run it
# from the repo root or from inside training/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINTS_CURRENT = os.path.join(_THIS_DIR, "..", "checkpoints", "current")

OLD_MODEL = os.path.join(_CHECKPOINTS_CURRENT, "pokemon_ppo_masked_v7")
NEW_MODEL = os.path.join(_CHECKPOINTS_CURRENT, "pokemon_ppo_masked_v7_expanded")

OLD_OBS_DIM = 285
NEW_OBS_DIM = 417
NUM_NEW_FEATURES = NEW_OBS_DIM - OLD_OBS_DIM

# Matches env.py's `self.action_space = spaces.Discrete(17)` -- this script
# never steps a real env, so it doesn't import/construct PokemonBattleEnv at
# all (that would require a live mGBA socket connection on port 8888, which
# this one-time weight-surgery operation doesn't need). If you ever change
# the action space size in env.py, update this constant to match.
NUM_ACTIONS = 17


class DummyBattleEnv(gym.Env):
    """Stand-in for PokemonBattleEnv with the same observation/action space
    shapes, but no socket connection to mGBA. Good enough for MaskablePPO.load()
    and MaskablePPO(...) construction, since neither call ever steps the env --
    they only read its space definitions to size the policy network."""

    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        # Bounds must match env.py's real Box exactly (low=0.0, high=1.0,
        # float32) -- SB3's check_for_correct_spaces() compares bounds and
        # dtype, not just shape, so a mismatch here fails model loading.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_actions)
        self._obs_dim = obs_dim

    def reset(self, *, seed=None, options=None):
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self._obs_dim, dtype=np.float32), 0.0, True, False, {}

    def action_masks(self):
        # All actions "legal" -- never actually used, since this env is
        # never stepped, but ActionMasker/MaskablePPO expect the method to exist.
        return np.ones(self.action_space.n, dtype=bool)


def mask_fn(env):
    return env.unwrapped.action_masks()


def expand_first_layer(old_layer: torch.nn.Linear) -> torch.nn.Linear:
    """Build a new Linear layer with NEW_OBS_DIM inputs. Copies the old
    weight columns (for the 285 inputs that still mean the same thing)
    unchanged, and randomly initializes only the new 132 columns."""
    out_features = old_layer.out_features
    new_layer = torch.nn.Linear(NEW_OBS_DIM, out_features)

    with torch.no_grad():
        # Old weights, unchanged, for the first 285 input columns.
        new_layer.weight[:, :OLD_OBS_DIM] = old_layer.weight.data
        # New random weights for the 132 newly appended columns -- PyTorch's
        # default Linear init already randomized new_layer on construction,
        # so columns [OLD_OBS_DIM:] are already randomly initialized; leave
        # them as-is (do not overwrite with zeros, which would make the new
        # features start out contributing nothing and slow to learn).
        new_layer.bias[:] = old_layer.bias.data

    return new_layer


def main():
    print(f"Loading old model '{OLD_MODEL}.zip' (trained on {OLD_OBS_DIM}-dim obs)...")
    # Dummy env, same OLD_OBS_DIM shape -- no mGBA/socket connection needed,
    # since we only read/rewrite the policy's weight tensors below, never
    # step the env.
    old_env = DummyBattleEnv(OLD_OBS_DIM, NUM_ACTIONS)
    old_env = ActionMasker(old_env, mask_fn)
    old_model = MaskablePPO.load(OLD_MODEL, env=old_env)

    print(f"Building new model shell with {NEW_OBS_DIM}-dim obs...")
    new_env = DummyBattleEnv(NEW_OBS_DIM, NUM_ACTIONS)  # 417-dim shape, matching the updated env.py
    new_env = ActionMasker(new_env, mask_fn)
    new_model = MaskablePPO(
        "MlpPolicy",
        new_env,
        learning_rate=old_model.learning_rate,
        ent_coef=old_model.ent_coef,
        clip_range=old_model.clip_range,
        batch_size=old_model.batch_size,
        n_steps=old_model.n_steps,
        gamma=old_model.gamma,
        verbose=1,
    )

    # Policy network: expand just the first layer, copy everything else.
    old_policy_sd = old_model.policy.state_dict()
    new_policy_sd = new_model.policy.state_dict()

    for key, old_tensor in old_policy_sd.items():
        if key not in new_policy_sd:
            print(f"  skip (no matching key in new model): {key}")
            continue

        new_tensor = new_policy_sd[key]
        if old_tensor.shape == new_tensor.shape:
            # Unchanged shape -- direct copy, no learning lost.
            new_policy_sd[key] = old_tensor.clone()
        elif old_tensor.dim() == 2 and old_tensor.shape[0] == new_tensor.shape[0] \
                and old_tensor.shape[1] == OLD_OBS_DIM and new_tensor.shape[1] == NEW_OBS_DIM:
            # This is the first layer's weight matrix -- expand it.
            print(f"  expanding first-layer weights: {key} {tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)}")
            expanded = new_tensor.clone()
            expanded[:, :OLD_OBS_DIM] = old_tensor
            new_policy_sd[key] = expanded
        else:
            print(f"  WARNING: shape mismatch, leaving randomly initialized: {key} "
                  f"old={tuple(old_tensor.shape)} new={tuple(new_tensor.shape)}")

    new_model.policy.load_state_dict(new_policy_sd)

    new_model.save(NEW_MODEL)
    print(f"\nDone. Saved '{NEW_MODEL}.zip'.")
    print("Point train_dqn.py's model_name at this file to resume training")
    print("with the old model's learning intact, plus fresh weights for the")
    print("new bench-effectiveness features.")

    old_env.close()
    new_env.close()


if __name__ == "__main__":
    main()
