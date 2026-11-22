import os
import numpy as np
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env import PokemonBattleEnv, OBS_DIM, N_ACTIONS, MOVE_TEACH, FRAME_STACK
import logger
from functools import partial
from stable_baselines3.common.vec_env import SubprocVecEnv
from check_ports import ports_from_env, check_ports, format_report

# --- Custom Win Rate Tracker Callback ---
class WinRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.battle_results = deque(maxlen=100) # Tracks win/loss for the last 100 battles
        self.battle_stats = {}                  # name -> deque of last 100 battles' values
        # (level_gap, won) for the last 300 decisive wild battles, so win rate can be
        # compared at a fixed matchup difficulty instead of a drifting average.
        self.gap_results = deque(maxlen=300)
        # (gap, won, fled) for wild battles that ended in a win, a loss or the player leaving: the win rate that
        # counts a flee as not-won, so an agent can't look better by escaping the hard fights.
        self.wild_results = deque(maxlen=300)
        self.trainer_results = deque(maxlen=100)   # trainer fights: tracked apart, their gaps mean something else

    def _on_step(self) -> bool:
        # Intercept the 'dones' and 'infos' array from the environment
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    # Only decisive battles count: a catch or a run is neither a
                    # win nor a loss. Fall back to player_lost if no outcome.
                    outcome = infos[i].get("outcome")
                    bs_all = infos[i].get("battle_stats") or {}
                    if outcome in (1, 2, 9, 4, 5) and bs_all.get("trainer_battle") == 0 and bs_all.get("level_gap") is not None:
                        self.wild_results.append((bs_all["level_gap"], 1.0 if outcome == 1 else 0.0,
                                                  1.0 if outcome in (4, 5) else 0.0))
                    if outcome in (1, 2, 9):
                        self.battle_results.append(1.0 if outcome == 1 else 0.0)
                        gap = (infos[i].get("battle_stats") or {}).get("level_gap")
                        if (infos[i].get("battle_stats") or {}).get("trainer_battle") == 1:
                            self.trainer_results.append(1.0 if outcome == 1 else 0.0)   # kept out of the wild gap buckets
                        elif gap is not None:
                            bs = infos[i].get("battle_stats") or {}
                            self.gap_results.append((gap, 1.0 if outcome == 1 else 0.0,
                                                     bs.get("turns", 0), bs.get("return", 0.0)))
                    elif outcome is None:
                        self.battle_results.append(0.0 if infos[i].get("player_lost", False) else 1.0)

        # Per-battle behaviour stats (voluntary switches, turns, ...) from the env
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                stats = infos[i].get("battle_stats") if done else None
                if stats:
                    for k, v in stats.items():
                        # Only numbers are averaged: battle_stats also carries text for the logs (species names, map),
                        # and one string in the deque used to crash np.mean at the first update.
                        if isinstance(v, (bool, int, float, np.integer, np.floating)) and v == v:   # v == v drops NaN
                            self.battle_stats.setdefault(k, deque(maxlen=100)).append(float(v))
        return True

    def _on_rollout_end(self) -> None:
        for k, q in self.battle_stats.items():
            if q:
                try:
                    self.logger.record(f"battle/{k}_mean", float(np.mean(q)))
                except (TypeError, ValueError):
                    pass          # a metrics problem must never stop a training run
        for name, lo, hi in (("le5", -999, 5), ("6_10", 6, 10), ("gt10", 11, 999)):
            w = [(r, t, ret) for g, r, t, ret in self.gap_results if lo <= g <= hi]
            if len(w) >= 10:
                self.logger.record(f"winrate_by_gap/{name}", float(np.mean([x[0] for x in w])) * 100.0)
                self.logger.record(f"turns_by_gap/{name}", float(np.mean([x[1] for x in w])))
                self.logger.record(f"return_by_gap/{name}", float(np.mean([x[2] for x in w])))
        for name, lo, hi in (("le5", -999, 5), ("6_10", 6, 10), ("gt10", 11, 999)):
            w = [(won, fled) for g, won, fled in self.wild_results if lo <= g <= hi]
            if len(w) >= 10:
                self.logger.record(f"winrate_all_by_gap/{name}", float(np.mean([x[0] for x in w])) * 100.0)
                self.logger.record(f"flee_rate_by_gap/{name}", float(np.mean([x[1] for x in w])) * 100.0)
        if len(self.trainer_results) >= 5:
            self.logger.record("winrate_trainer", float(np.mean(self.trainer_results)) * 100.0)
        # Calculate win rate and push it to the terminal table and TensorBoard
        if len(self.battle_results) > 0:
            win_rate = np.mean(self.battle_results) * 100.0
            self.logger.record("rollout/win_rate_pct", win_rate)


class EntCoefAnneal(BaseCallback):
    """Linearly lowers ent_coef from `start` to `end` over `total` timesteps (opt-in: POKE_ENT_ANNEAL=1).
    The policy is sampled in training but usually judged deterministically, so a lower late entropy bonus
    sharpens it once the easy gains are made. Progress counts from where this run started."""
    def __init__(self, start=0.01, end=0.002, total=500_000, verbose=0):
        super().__init__(verbose)
        self.start, self.end, self.total = start, end, max(int(total), 1)
        self.t0 = None

    def value(self, done):
        frac = min(max(done / self.total, 0.0), 1.0)
        return self.start + (self.end - self.start) * frac

    def _on_training_start(self) -> None:
        self.t0 = self.model.num_timesteps
        self.model.ent_coef = self.value(0)

    def _on_rollout_end(self) -> None:
        self.model.ent_coef = self.value(self.model.num_timesteps - (self.t0 or 0))
        self.logger.record("train/ent_coef_sched", float(self.model.ent_coef))

    def _on_step(self) -> bool:
        return True


def save_atomic(model, path):
    """model.save(path) without ever leaving a half-written file at `path` (write beside it, then rename)."""
    tmp = f"{path}.tmp.zip"               # ends in .zip, so SB3 writes exactly this name
    model.save(tmp)
    os.replace(tmp, f"{path}.zip")


class SafetyCheckpointCallback(BaseCallback):
    """Crash safety: every `save_freq` calls (one call = one step of all envs) rewrite backups/latest_safety and, when
    `main_path` is given, the main model file too. The main file is what the next run loads, so progress survives a
    Ctrl+C or a crash instead of only existing in backups/."""
    def __init__(self, save_freq=256, save_path="./backups/", main_path=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.main_path = main_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.save_path, "latest_safety"))
            if self.main_path:
                save_atomic(self.model, self.main_path)
        return True


def model_suffix(frame_stack, move_teach):
    """Checkpoint name suffix for the effective observation/action layout (matches expand_model_v9.py)."""
    return ("_stack" if frame_stack else "") + ("_teach" if move_teach else "")


# --- Masking Function ---
def mask_fn(env):
    return env.unwrapped.action_masks()


EPISODE_BLOCK = 10_000_000       # env i numbers its battles from i * EPISODE_BLOCK, so episode ids are unique in every log


def ppo_settings(environ=None, n_envs=1):
    """PPO update settings from POKE_N_STEPS (total samples per update across all envs), POKE_BATCH_SIZE, POKE_N_EPOCHS and
    POKE_TARGET_KL. One real emulator yields ~1,000 steps an hour, so the old 2,048-step update took ~2 hours; the default
    is now 512 / 128 (about 75 battles per update) with a KL early stop. v8 used POKE_N_STEPS=2048 POKE_BATCH_SIZE=512."""
    env = os.environ if environ is None else environ
    total = max(int(env.get("POKE_N_STEPS", 512)), 32)
    n_steps = max(total // max(n_envs, 1), 32)            # SB3's n_steps is per env
    samples = n_steps * max(n_envs, 1)
    batch = max(min(int(env.get("POKE_BATCH_SIZE", 128)), samples), 1)
    return {"n_steps": n_steps, "batch_size": batch, "n_epochs": int(env.get("POKE_N_EPOCHS", 10)),
            "target_kl": float(env.get("POKE_TARGET_KL", 0.03)), "samples_per_update": samples}


def make_env(port, idx, run_id, multi):
    """One training env on its own bridge port. Top-level so SubprocVecEnv can pickle it (Python 3.14 starts workers with
    forkserver). In a multi-env run each worker gets its own log id and episode numbering."""
    env = PokemonBattleEnv(port=port)
    if multi:
        logger.set_env(f"{run_id}_e{idx}", f"e{idx}")
        env.episode = idx * EPISODE_BLOCK
    return ActionMasker(env, mask_fn)


if __name__ == "__main__":
    # 0. Logs are created once here, so workers only append. One bridge port per env (POKE_N_ENVS windows).
    n_envs = max(int(os.environ.get("POKE_N_ENVS", 1)), 1)
    ports = ports_from_env(n_envs)
    logger.prepare_logs()
    _suffix = model_suffix(FRAME_STACK, MOVE_TEACH)       # effective settings, not raw variables (legacy mode turns both off)
    run_id = logger.set_run("train_v9" + _suffix)

    # 1. Initialize the environment(s) and apply the action mask wrapper
    if n_envs == 1:
        env = make_env(ports[0], 0, run_id, False)
    else:
        text, bad = format_report(check_ports(ports))
        print(text)
        if bad:
            raise SystemExit(f"{bad} of {n_envs} bridges did not answer; load games/battle_bridge.lua in each window "
                             "(first window first) and check with: python training/check_ports.py")
        env = SubprocVecEnv([partial(make_env, p, i, run_id, True) for i, p in enumerate(ports)])
    cfg = ppo_settings(os.environ, n_envs)
    print(f"PPO: {n_envs} env(s) on ports {ports}; {cfg['n_steps']} steps x {n_envs} = {cfg['samples_per_update']} samples "
          f"per update, batch {cfg['batch_size']}, {cfg['n_epochs']} epochs, target_kl {cfg['target_kl']}")

    # Absolute path, so it loads (and saves) the same checkpoint no matter which
    # folder training is launched from.
    _ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "current")
    base_name = os.path.join(_ckpt, "pokemon_ppo_masked_v8")    # baseline: loaded, never overwritten
    model_name = os.path.join(_ckpt, "pokemon_ppo_masked_v9" + _suffix)   # this run saves here

    # 2. Setup the automated milestone backups (numbered, accumulating --
    #    saved once per update: SB3 counts save_freq in calls, one call per step of all envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["n_steps"],
        save_path="./backups/",
        name_prefix="ppo_v9" + _suffix       # v8's milestones are ppo_model_<n>_steps; v9 counts from 0 and must not overwrite them
    )

    # 3. Setup the custom Win Rate table tracker
    win_rate_callback = WinRateCallback()

    # 3b. Setup the crash-safety net (single overwriting file, every 256 steps)
    safety_checkpoint = SafetyCheckpointCallback(save_freq=max(256 // n_envs, 8), save_path="./backups/", main_path=model_name)

    callbacks = [checkpoint_callback, win_rate_callback, safety_checkpoint]
    if os.environ.get("POKE_ENT_ANNEAL") == "1":
        callbacks.append(EntCoefAnneal(start=0.01, end=0.002, total=500_000))

    # 4. Load weights or start fresh
    load_name = model_name if os.path.exists(f"{model_name}.zip") else base_name
    if load_name == base_name and env.observation_space.shape[0] != 431 and not os.path.exists(f"{model_name}.zip"):
        raise SystemExit(f"{os.path.basename(model_name)}.zip not found. Build it first: python training/expand_model_v9.py"
                         + (" --stack" if FRAME_STACK else "") + (" --teach" if MOVE_TEACH else ""))
    if os.path.exists(f"{load_name}.zip"):
        print(f"Loading existing model '{load_name}.zip'...")
        model = MaskablePPO.load(load_name, env=env, tensorboard_log="./ppo_tensorboard/", n_steps=cfg["n_steps"],
                                 batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"], target_kl=cfg["target_kl"])
        # Fine-tune under the new reward: the saved ent_coef (0.05) keeps the
        model.ent_coef = 0.01
    else:
        print("Creating a new model...")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            ent_coef=0.05,        # Anti-bias exploration
            clip_range=0.2,
            batch_size=cfg["batch_size"],
            n_steps=cfg["n_steps"],
            n_epochs=cfg["n_epochs"],
            target_kl=cfg["target_kl"],
            gamma=0.995,          # Focuses on long-term wins
            vf_coef=1.0,
            verbose=1,
            tensorboard_log="./ppo_tensorboard/"
        )

    logger.log_run_start("train_dqn.py", OBS_DIM, N_ACTIONS,
                         (os.path.basename(load_name) if os.path.exists(f"{load_name}.zip") else "(new model)")
                         + f" | envs={n_envs} ports={','.join(map(str, ports))} n_steps={cfg['n_steps']} batch={cfg['batch_size']}")
    print(f"Run id: {logger.RUN_ID}")
    # How many MORE samples to train for (SB3 adds this to the count already in the loaded model). Not tied to the
    # update size: a 512-sample update is one learning step inside this run, not the end of it.
    total_steps = int(os.environ.get("POKE_TOTAL_STEPS", 500_000))
    print(f"Training for {total_steps:,} more samples (POKE_TOTAL_STEPS), updating every {cfg['samples_per_update']}.")
    print("Starting FULL protected training run...")

    # 5. Run the massive training loop with all callbacks active
    try:
        model.learn(
            total_timesteps=total_steps,
            tb_log_name="run_v9",
            callback=callbacks,
            reset_num_timesteps=False
        )
        print(f"\nTraining reached the {total_steps:,} step finish line!")
    except KeyboardInterrupt:
        print("\nStopped by you (Ctrl+C): saving the model before exit...")
    finally:
        save_atomic(model, model_name)          # Ctrl+C, a crash or the finish line: the latest weights are in the main file
        try:
            env.close()
        except Exception:
            pass
        print(f"Model saved as {model_name}.zip ({model.num_timesteps:,} samples so far)")
