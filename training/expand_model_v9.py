"""One-time: carry the v8 checkpoint over to the v9 env (499-dim obs; +61 with --stack, +44 and 4 more actions with --teach).

v9 only APPENDS 60 combat + 8 opponent-team features (and 61 frame-stack features with --stack) (damage, KO flags, speed, stats, bench) after v8's 431 inputs, so
every old weight keeps its meaning. New input columns start at zero, so the expanded model behaves
exactly like v8 until training grows them. The value head is copied unchanged (see
analysis/reinit_value_head.py if explained_variance stays ~0). No emulator needed:

    python training/expand_model_v9.py           # 499-value obs (default env)
    python training/expand_model_v9.py --stack   # 560-value obs, for POKE_FRAME_STACK=1
    python training/expand_model_v9.py --teach   # 543-value obs and 22 actions, for POKE_MOVE_TEACH=1 (combine with --stack: 604)
"""
import os
import sys
import torch
from sb3_contrib import MaskablePPO

from expand_model_v8 import CUR, Dummy, wrap

STACK, TEACH = "--stack" in sys.argv, "--teach" in sys.argv

OLD_MODEL = os.path.join(CUR, "pokemon_ppo_masked_v8")
NEW_MODEL = os.path.join(CUR, "pokemon_ppo_masked_v9" + ("_stack" if STACK else "") + ("_teach" if TEACH else ""))
OLD_OBS, OLD_ACT = 431, 18
NEW_OBS = 499 + (61 if STACK else 0) + (44 if TEACH else 0)     # block order: combat, trainer, stack, teach
N_ACT = 18 + (4 if TEACH else 0)
NEW_ACTION_BIAS = -2.0       # new teach actions start rare; their shaping reward teaches the policy when they pay


def main():
    old = MaskablePPO.load(OLD_MODEL, env=wrap(Dummy(OLD_OBS, OLD_ACT)))
    new = MaskablePPO(
        "MlpPolicy", wrap(Dummy(NEW_OBS, N_ACT)),
        learning_rate=old.learning_rate, ent_coef=old.ent_coef, clip_range=old.clip_range,
        batch_size=old.batch_size, n_steps=old.n_steps, gamma=old.gamma,
        vf_coef=old.vf_coef, verbose=1)
    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
    for key, o in old_sd.items():
        n = new_sd.get(key)
        if n is None:
            print(f"skip (not in new model): {key}")
        elif o.shape == n.shape:
            new_sd[key] = o.clone()
        elif o.dim() == 2 and o.shape[1] == OLD_OBS and n.shape[1] == NEW_OBS and o.shape[0] == n.shape[0]:
            print(f"expand inputs {key}: {tuple(o.shape)} -> {tuple(n.shape)}")
            e = torch.zeros_like(n)          # zero: v9 starts out identical to v8
            e[:, :OLD_OBS] = o
            new_sd[key] = e
        elif key.startswith("action_net") and o.shape[0] == OLD_ACT and n.shape[0] == N_ACT:
            print(f"expand actions {key}: {tuple(o.shape)} -> {tuple(n.shape)}")
            e = torch.zeros_like(n)          # zero weights; the bias keeps the new actions improbable at first
            if o.dim() == 1:
                e[OLD_ACT:] = NEW_ACTION_BIAS
            e[:OLD_ACT] = o
            new_sd[key] = e
        else:
            print(f"WARNING shape mismatch, left random: {key} {tuple(o.shape)} vs {tuple(n.shape)}")
    new.policy.load_state_dict(new_sd)
    new.save(NEW_MODEL)
    print(f"Saved {NEW_MODEL}.zip")


if __name__ == "__main__":
    main()
