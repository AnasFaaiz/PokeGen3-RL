"""Offline checks for parallel training and the PPO settings: python training/test_parallel.py"""
import os
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_ports as cp
import prepare_instances as pi
import train_dqn as t

# --- ports
assert cp.ports_from_env(4, {}) == [8888, 8890, 8892, 8894]
assert cp.ports_from_env(2, {"POKE_PORT_BASE": "9000", "POKE_PORT_STEP": "10"}) == [9000, 9010]
assert cp.ports_from_env(3, {"POKE_PORTS": "8888, 8892,8896"}) == [8888, 8892, 8896]
try:
    cp.ports_from_env(2, {"POKE_PORTS": "8888"})
    raise AssertionError("a port list of the wrong length must be rejected")
except ValueError:
    pass

# --- a fake bridge that answers STATUS and INVENTORY, then closes (like check_ports does)
class FakeBridge(threading.Thread):
    def __init__(self, new_lua=True, bad_status=False):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(2)
        self.port = self.srv.getsockname()[1]
        self.new_lua, self.bad_status = new_lua, bad_status

    def run(self):
        try:
            conn, _ = self.srv.accept()
        except OSError:
            return
        f = conn.makefile("rw")
        for line in f:
            line = line.strip()
            if line == "STATUS":
                f.write("garbage\n" if self.bad_status else "STATUS:CHOOSE_ACTION:NONE\n")
            elif line == "INVENTORY":
                f.write("INV:money=0;balls=4:10;party=6;lv=20,20,20,20,20,20;outcome=0;trainer=0;map=0.32"
                        + (";px=1:100" if self.new_lua else "") + "\n")
            f.flush()
        conn.close()

b1, b2, b3 = FakeBridge(), FakeBridge(new_lua=False), FakeBridge(bad_status=True)
for b in (b1, b2, b3):
    b.start()
dead = socket.socket(); dead.bind(("127.0.0.1", 0)); dead_port = dead.getsockname()[1]; dead.close()   # nothing listens here
res = cp.check_ports([b1.port, b2.port, b3.port, dead_port], timeout=2.0)
assert [r[1] for r in res] == [True, True, False, False], res
assert res[0][2] == {"phase": "CHOOSE_ACTION", "map": "0.32", "party": "6", "has_px": True} and res[1][2]["has_px"] is False
text, bad = cp.format_report(res)
assert bad == 2 and "reload the script" in text and "no answer" in text and "unexpected reply" in text, text

# --- PPO settings (one emulator: 512 total; several envs split the total; v8's cadence is one override away)
d = t.ppo_settings({}, 1)
assert (d["n_steps"], d["batch_size"], d["n_epochs"], d["target_kl"], d["samples_per_update"]) == (512, 128, 10, 0.03, 512)
d = t.ppo_settings({}, 4)
assert d["n_steps"] == 128 and d["samples_per_update"] == 512 and d["batch_size"] == 128
d = t.ppo_settings({"POKE_N_STEPS": "2048", "POKE_BATCH_SIZE": "512"}, 1)
assert (d["n_steps"], d["batch_size"]) == (2048, 512)
d = t.ppo_settings({"POKE_N_STEPS": "256"}, 16)                              # per-env minimum 32
assert d["n_steps"] == 32 and d["samples_per_update"] == 512
d = t.ppo_settings({"POKE_N_STEPS": "256", "POKE_BATCH_SIZE": "1024"}, 1)    # batch never exceeds the samples per update
assert d["batch_size"] == 256
for envs in (1, 2, 4, 8):
    s = t.ppo_settings({}, envs)
    assert s["batch_size"] <= s["samples_per_update"] and s["samples_per_update"] % s["batch_size"] == 0, (envs, s)

# --- SB3 re-creates the rollout buffer when the settings are overridden on load (the v8 checkpoint is only read)
from expand_model_v8 import Dummy, wrap
from sb3_contrib import MaskablePPO
v8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "current", "pokemon_ppo_masked_v8")
if os.path.exists(v8 + ".zip"):
    m = MaskablePPO.load(v8, env=wrap(Dummy(431, 18)), n_steps=128, batch_size=64, n_epochs=3, target_kl=0.03)
    assert (m.n_steps, m.batch_size, m.n_epochs, m.target_kl) == (128, 64, 3, 0.03) and m.rollout_buffer.buffer_size == 128

# --- the env factory pickles (SubprocVecEnv workers need it) and episode numbering is unique per env
pickle.dumps(partial(t.make_env, 8890, 1, "run_x", True))
blocks = [range(i * t.EPISODE_BLOCK, (i + 1) * t.EPISODE_BLOCK) for i in range(16)]
assert all(b1_.stop <= b2_.start for b1_, b2_ in zip(blocks, blocks[1:])) and t.EPISODE_BLOCK >= 1_000_000

# --- 2 env processes feed one MaskablePPO through action masks (dummy env; the real one needs an emulator)
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv


class MiniEnv(gym.Env):
    def __init__(self, idx):
        self.observation_space = spaces.Box(0.0, 1.0, (6,), np.float32)
        self.action_space = spaces.Discrete(5)
        self.idx, self.n = idx, 0

    def reset(self, *, seed=None, options=None):
        self.n = 0
        return np.full(6, self.idx / 4.0, np.float32), {}

    def step(self, a):
        self.n += 1
        return np.full(6, self.idx / 4.0, np.float32), float(a == 2), self.n >= 3, False, {"idx": self.idx}

    def action_masks(self):
        m = np.ones(5, bool)
        m[4] = False                       # action 4 is never legal: the policy must respect the mask
        return m


def make_mini(idx):
    return ActionMasker(MiniEnv(idx), lambda e: e.unwrapped.action_masks())


if __name__ == "__main__":
    vec = SubprocVecEnv([partial(make_mini, i) for i in range(2)])
    model = MaskablePPO("MlpPolicy", vec, n_steps=16, batch_size=16, n_epochs=2, verbose=0)
    assert model.rollout_buffer.buffer_size == 16 and model.n_envs == 2
    model.learn(64)
    acts = [int(model.predict(np.zeros(6, np.float32), action_masks=np.array([[1, 1, 1, 1, 0]], bool))[0]) for _ in range(20)]
    assert 4 not in acts
    vec.close()

    # --- prepare_instances: one folder per window, own save copy, no silent overwrite, printed commands only
    with tempfile.TemporaryDirectory() as d:
        rom = os.path.join(d, "Emerald.gba"); open(rom, "wb").write(b"ROM")
        open(os.path.join(d, "Emerald.sav"), "wb").write(b"SAVE")
        state = os.path.join(d, "Emerald.ss1"); open(state, "wb").write(b"STATE")
        out = os.path.join(d, "instances")
        plan = pi.make_plan(rom, 3, state, out)
        assert [p["index"] for p in plan] == [0, 1, 2] and len({p["dir"] for p in plan}) == 3
        assert all(p["command"].startswith("flatpak run io.mgba.mGBA -t ") and p["command"].endswith("Emerald.gba") for p in plan)
        notes = pi.copy_instances(rom, plan, state)
        for p in plan:
            assert open(p["rom"], "rb").read() == b"ROM" and open(p["sav"], "rb").read() == b"SAVE" and open(p["state"], "rb").read() == b"STATE"
        assert notes == []
        open(plan[1]["sav"], "wb").write(b"HALF-PLAYED")                       # a save that moved on is kept, not replaced
        notes = pi.copy_instances(rom, plan, state)
        assert open(plan[1]["sav"], "rb").read() == b"HALF-PLAYED" and any("kept the existing save" in n for n in notes)
        assert not os.path.exists(plan[0]["sav"] + ".tmp")
        os.remove(os.path.join(d, "Emerald.sav"))                              # no save next to the ROM: say so
        assert any("no Emerald.sav" in n for n in pi.copy_instances(rom, pi.make_plan(rom, 1, None, os.path.join(d, "x")), None))
        assert pi.make_plan(rom, 1, None, out)[0]["command"] == "flatpak run io.mgba.mGBA " + pi.make_plan(rom, 1, None, out)[0]["rom"]
        r = subprocess.run([sys.executable, pi.__file__, "--rom", os.path.join(d, "missing.gba"), "--n", "2"], capture_output=True, text=True)
        assert r.returncode != 0 and "ROM not found" in r.stderr + r.stdout

    # --- Lua port selection: first free port in the sequence, none when all are busy
    lua = shutil.which("lua5.4") or shutil.which("lua")
    if lua:
        ports_lua = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games", "bridge_ports.lua")
        code = f'''
local P = dofile("{ports_lua}")
local busy = {{}}
local function bind(_, port) if busy[port] then return nil end return {{port = port}} end
local s, p = P.bindFirst(bind, 8888, 2, 16); print(p)
busy[8888] = true; s, p = P.bindFirst(bind, 8888, 2, 16); print(p)
busy[8890] = true; busy[8892] = true; s, p = P.bindFirst(bind, 8888, 2, 16); print(p)
for i = 0, 15 do busy[8888 + 2 * i] = true end
s, p = P.bindFirst(bind, 8888, 2, 16); print(s, p)
'''
        out = subprocess.run([lua, "-e", code], capture_output=True, text=True)
        assert out.stdout.split() == ["8888", "8890", "8894", "nil", "nil"], out
    print("ok")
