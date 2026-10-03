import numpy as np
import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("battle_data.csv")

parsed = [json.loads(obs_str) for obs_str in df["obs_json"]]
df["obs_parsed"] = parsed
df["obs_len"] = [len(p) for p in parsed]

df = df[df["obs_len"] == 52]

print(f"Episode range in 52-dim data: {df['episode'].min()} to {df['episode'].max()}")

MIN_EPISODE = 20  # adjust based on the printed range
df_recent = df[df["episode"] >= MIN_EPISODE]

print(f"Total 52-dim rows: {len(df)}")
print(f"Recent rows (episode >= {MIN_EPISODE}): {len(df_recent)}")

if len(df_recent) == 0:
    print("No rows matched — adjust MIN_EPISODE based on the range printed above.")
else:
    X = np.array(df_recent["obs_parsed"].tolist())
    y = df_recent["action"].to_numpy()

    tree = DecisionTreeClassifier(max_depth=4)
    tree.fit(X, y)

    feature_names = [
        "player_hp", "enemy_hp",
        "pp1", "pp2", "pp3", "pp4",
        "move1", "move2", "move3", "move4",
        "move1_type", "move2_type", "move3_type", "move4_type",
        "move1_eff", "move2_eff", "move3_eff", "move4_eff",
        "move1_power", "move2_power", "move3_power", "move4_power",
        "move1_acc", "move2_acc", "move3_acc", "move4_acc",
        "move1_class", "move2_class", "move3_class", "move4_class",
        "p_type1", "p_type2", "e_type1", "e_type2",
        "p_level", "e_level",
        "p_status1", "e_status1", "p_status2", "e_status2",
        "p_atk_buff", "p_def_buff", "p_spd_buff",
        "e_atk_buff", "e_def_buff", "e_spd_buff",
        "party1_hp", "party2_hp", "party3_hp", "party4_hp", "party5_hp", "party6_hp",
    ]
    print(export_text(tree, feature_names=feature_names))

    accuracy = tree.score(X, y)
    print(f"\nTree matches DQN's RECENT real decisions {accuracy*100:.1f}% of the time")
    print(f"Based on {len(df_recent)} logged turns from episodes {MIN_EPISODE}+")
