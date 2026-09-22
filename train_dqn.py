from stable_baselines3 import DQN
from env import PokemonBattleEnv
import os 

env = PokemonBattleEnv()

# Changed to v2 to prevent crashing against the old 6-input model
model_name = "pokemon_dqn_v2" 

if os.path.exists(f"{model_name}.zip"):
    print("Loading existing model to continue training...")
    model = DQN.load(model_name, env=env)
else:
    print("No existing model found, creating a new one!")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=5000,
        learning_starts=50,
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/",
    )

# 200 timesteps is perfect for a quick sanity check to ensure no crashes!
model.learn(total_timesteps=500, tb_log_name="run_v2", reset_num_timesteps=False)
print(f"Total episodes completed: {env.episode}")
model.save(model_name)
env.close()
print(f"Training complete, model saved as {model_name}")
