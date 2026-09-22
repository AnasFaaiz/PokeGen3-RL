import numpy as np
from stable_baselines3 import DQN
from sklearn.tree import DecisionTreeClassifier, export_text
from env import PokemonBattleEnv

env = PokemonBattleEnv()
model = DQN.load("pokemon_dqn_v2", env=env)

X = []
y = []

for _ in range(1000):
    sample_obs = env.observation_space.sample()
    action, _ = model.predict(sample_obs, deterministic=True)
    X.append(sample_obs)
    y.append(action)

X = np.array(X)
y = np.array(y)

tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X, y)

feature_names = [
    "player_hp", "enemy_hp",
    "pp1", "pp2", "pp3", "pp4",
    "move1", "move2", "move3", "move4",
    "p_type1", "p_type2", "e_type1", "e_type2",
    "p_level", "e_level",
    "p_atk_buff", "p_def_buff", "p_spd_buff",
    "e_atk_buff", "e_def_buff", "e_spd_buff",
]
print(export_text(tree, feature_names=feature_names))

accuracy = tree.score(X, y)
print(f"\nTree matches DQN's decisions {accuracy*100:.1f}% of the time")

env.close()
