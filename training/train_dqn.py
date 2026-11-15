import os
import numpy as np
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env import PokemonBattleEnv

# --- Custom Win Rate Tracker Callback ---
class WinRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.battle_results = deque(maxlen=100) # Tracks win/loss for the last 100 battles
        self.battle_stats = {}                  # name -> deque of last 100 battles' values

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
                    if outcome in (1, 2, 9):
                        self.battle_results.append(1.0 if outcome == 1 else 0.0)
                    elif outcome is None:
                        self.battle_results.append(0.0 if infos[i].get("player_lost", False) else 1.0)

        # Per-battle behaviour stats (voluntary switches, turns, ...) from the env
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                stats = infos[i].get("battle_stats") if done else None
                if stats:
                    for k, v in stats.items():
                        self.battle_stats.setdefault(k, deque(maxlen=100)).append(v)
        return True

    def _on_rollout_end(self) -> None:
        for k, q in self.battle_stats.items():
            if q:
                self.logger.record(f"battle/{k}_mean", float(np.mean(q)))
        # Calculate win rate and push it to the terminal table and TensorBoard
        if len(self.battle_results) > 0:
            win_rate = np.mean(self.battle_results) * 100.0
            self.logger.record("rollout/win_rate_pct", win_rate)


class SafetyCheckpointCallback(BaseCallback):
    def __init__(self, save_freq=256, save_path="./backups/", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.save_path, "latest_safety"))
        return True


# --- Masking Function ---
def mask_fn(env):
    return env.unwrapped.action_masks()


if __name__ == "__main__":
    # 1. Initialize environment and apply the action mask wrapper
    env = PokemonBattleEnv()
    env = ActionMasker(env, mask_fn)

    # Absolute path, so it loads (and saves) the same checkpoint no matter which
    # folder training is launched from.
    model_name = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "current",
        "pokemon_ppo_masked_v7_expanded")

    # 2. Setup the automated milestone backups (numbered, accumulating --
    #    saved exactly when the brain updates, i.e. every n_steps=2048)
    checkpoint_callback = CheckpointCallback(
        save_freq=2048,
        save_path="./backups/",
        name_prefix="ppo_model"
    )

    # 3. Setup the custom Win Rate table tracker
    win_rate_callback = WinRateCallback()

    # 3b. Setup the crash-safety net (single overwriting file, every 256 steps)
    safety_checkpoint = SafetyCheckpointCallback(save_freq=256, save_path="./backups/")

    # 4. Load weights or start fresh
    if os.path.exists(f"{model_name}.zip"):
        print(f"Loading existing model '{model_name}.zip'...")
        model = MaskablePPO.load(model_name, env=env)
        # Fine-tune under the new reward: the saved ent_coef (0.05) keeps the
        # policy near-uniform, so lower it to let it commit to what pays off.
        model.ent_coef = 0.01
    else:
        print("Creating a new model...")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            ent_coef=0.05,        # Anti-bias exploration
            clip_range=0.2,
            batch_size=512,
            n_steps=2048,
            gamma=0.995,          # Focuses on long-term wins
            verbose=1,
            tensorboard_log="./ppo_tensorboard/"
        )

    print("Starting FULL protected training run...")

    # 5. Run the massive training loop with all callbacks active
    model.learn(
        total_timesteps=500_000,
        tb_log_name="run_v7",
        callback=[checkpoint_callback, win_rate_callback, safety_checkpoint],
        reset_num_timesteps=False
    )

    print("\nTraining reached the 500,000 step finish line!")
    model.save(model_name)
    env.close()
    print(f"Final model saved as {model_name}.zip")
