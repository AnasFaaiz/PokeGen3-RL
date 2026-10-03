import requests
import json
import time
import certifi

MAX_GEN3_MOVE_ID = 354

# Load your existing data
with open("emerald_moves.json") as f:
    moves_dict = json.load(f)

print("Fetching damage class (Physical/Special/Status) for all moves...")

for move_id_str in list(moves_dict.keys()):
    move_id = int(move_id_str)
    url = f"https://pokeapi.co/api/v2/move/{move_id}/"
    response = requests.get(url, verify=certifi.where())

    if response.status_code == 200:
        data = response.json()
        damage_class_name = data["damage_class"]["name"]  # "physical", "special", or "status"

        # Encode as an integer for the observation space:
        # 0 = status, 1 = physical, 2 = special
        damage_class_map = {"status": 0, "physical": 1, "special": 2}
        damage_class_id = damage_class_map.get(damage_class_name, 0)

        moves_dict[move_id_str]["damage_class"] = damage_class_name
        moves_dict[move_id_str]["damage_class_id"] = damage_class_id
    else:
        print(f"Failed to fetch move {move_id}, skipping")

    time.sleep(0.1)

    if move_id % 50 == 0:
        print(f"Updated {move_id} / {MAX_GEN3_MOVE_ID} moves...")

# Save back to the same file, now enriched
with open("emerald_moves.json", "w") as f:
    json.dump(moves_dict, f, indent=4)

print("Successfully added damage_class to all moves in emerald_moves.json!")
