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

    def _on_step(self) -> bool:
        # Intercept the 'dones' and 'infos' array from the environment
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    player_lost = infos[i].get("player_lost", False)
                    self.battle_results.append(0.0 if player_lost else 1.0)
        return True

    def _on_rollout_end(self) -> None:
        # Calculate win rate and push it to the terminal table and TensorBoard
        if len(self.battle_results) > 0:
            win_rate = np.mean(self.battle_results) * 100.0
            self.logger.record("rollout/win_rate_pct", win_rate)

# --- Masking Function ---
def mask_fn(env):
    return env.unwrapped.action_masks()


if __name__ == "__main__":
    # 1. Initialize environment and apply the action mask wrapper
    env = PokemonBattleEnv()
    env = ActionMasker(env, mask_fn)
    
    model_name = "pokemon_ppo_masked_v7"

    # 2. Setup the automated backups (saves exactly when the brain updates)
    checkpoint_callback = CheckpointCallback(
        save_freq=2048,
        save_path="./backups/",
        name_prefix="ppo_model"
    )

    # 3. Setup the custom Win Rate table tracker
    win_rate_callback = WinRateCallback()

    # 4. Load weights or start fresh
    if os.path.exists(f"{model_name}.zip"):
        print(f"Loading existing model '{model_name}.zip'...")
        model = MaskablePPO.load(model_name, env=env)
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

    # 5. Run the massive training loop with both callbacks active
    model.learn(
        total_timesteps=500_000, 
        tb_log_name="run_v7", 
        callback=[checkpoint_callback, win_rate_callback], 
        reset_num_timesteps=False
    )

    print("\nTraining reached the 500,000 step finish line!")
    model.save(model_name)
    env.close()
    print(f"Final model saved as {model_name}.zip")
