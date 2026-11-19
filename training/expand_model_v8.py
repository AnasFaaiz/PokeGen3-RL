"""One-time: carry the v7 checkpoint over to the v8 env (431-dim obs, 18 actions).

v8 only APPENDS to the old layout (14 obs features after index 416, action 17
after action 16), so every old weight still means what it did. This copies them
and adds fresh weights only for the new inputs / the new action. The value head
is re-initialised because rewards are now scaled by REWARD_SCALE (and the level
and new-species terms are new), so the old critic's outputs are on the wrong
scale. No emulator needed:

    python expand_model_v8.py
"""
import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

HERE = os.path.dirname(os.path.abspath(__file__))
CUR = os.path.join(HERE, "..", "checkpoints", "current")
OLD_MODEL = os.path.join(CUR, "pokemon_ppo_masked_v7_expanded")
NEW_MODEL = os.path.join(CUR, "pokemon_ppo_masked_v8")
OLD_OBS, NEW_OBS, OLD_ACT, NEW_ACT = 417, 431, 17, 18


class Dummy(gym.Env):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
        self.action_space = spaces.Discrete(n_act)
        self._d = obs_dim

    def reset(self, *, seed=None, options=None):
        return np.zeros(self._d, np.float32), {}

    def step(self, a):
        return np.zeros(self._d, np.float32), 0.0, True, False, {}

    def action_masks(self):
        return np.ones(self.action_space.n, dtype=bool)


def wrap(env):
    return ActionMasker(env, lambda e: e.unwrapped.action_masks())


def main():
    old = MaskablePPO.load(OLD_MODEL, env=wrap(Dummy(OLD_OBS, OLD_ACT)))
    new = MaskablePPO(
        "MlpPolicy", wrap(Dummy(NEW_OBS, NEW_ACT)),
        learning_rate=old.learning_rate, ent_coef=old.ent_coef, clip_range=old.clip_range,
        batch_size=old.batch_size, n_steps=old.n_steps, gamma=old.gamma,
        vf_coef=1.0, verbose=1)

    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
    for key, o in old_sd.items():
        n = new_sd.get(key)
        if n is None:
            print(f"skip (not in new model): {key}")
        elif key.startswith("value_net"):
            print(f"re-init value head (reward scale changed): {key}")
            new_sd[key] = torch.zeros_like(n)
        elif o.shape == n.shape:
            new_sd[key] = o.clone()
        elif o.dim() == 2 and o.shape[1] == OLD_OBS and n.shape[1] == NEW_OBS and o.shape[0] == n.shape[0]:
            print(f"expand inputs {key}: {tuple(o.shape)} -> {tuple(n.shape)}")
            e = n.clone()
            e[:, :OLD_OBS] = o
            new_sd[key] = e
        elif key.startswith("action_net") and o.shape[0] == OLD_ACT and n.shape[0] == NEW_ACT:
            print(f"expand actions {key}: {tuple(o.shape)} -> {tuple(n.shape)}")
            e = torch.zeros_like(n) if o.dim() == 1 else n.clone() * 0.01
            e[:OLD_ACT] = o
            new_sd[key] = e
        else:
            print(f"WARNING shape mismatch, left random: {key} {tuple(o.shape)} vs {tuple(n.shape)}")
    new.policy.load_state_dict(new_sd)
    new.save(NEW_MODEL)
    print(f"Saved {NEW_MODEL}.zip")


if __name__ == "__main__":
    main()
