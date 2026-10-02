import requests
import json
import time
import certifi

# Gen 3 includes moves from ID 1 (Pound) to 354 (Psycho Boost)
MAX_GEN3_MOVE_ID = 354

# We map types to integers so your DQN observation space can process them
TYPE_MAP = {
    "normal": 0, "fighting": 1, "flying": 2, "poison": 3, "ground": 4,
    "rock": 5, "bug": 6, "ghost": 7, "steel": 8, "fire": 9, "water": 10,
    "grass": 11, "electric": 12, "psychic": 13, "ice": 14, "dragon": 15,
    "dark": 16, "unknown": 17
}

moves_dict = {}

print("Fetching Gen 3 Move Data from PokeAPI...")

for move_id in range(1, MAX_GEN3_MOVE_ID + 1):
    url = f"https://pokeapi.co/api/v2/move/{move_id}/"
    response = requests.get(url, verify=certifi.where())

    if response.status_code == 200:
        data = response.json()
        
        # Clean up data for Neural Network ingestion
        # 1. Status moves have 'None' power. We convert to 0.
        power = data["power"] if data["power"] is not None else 0
        
        # 2. Moves that never miss (e.g., Swift) have 'None' accuracy. We convert to 100.
        accuracy = data["accuracy"] if data["accuracy"] is not None else 100
        
        # 3. Convert string type to integer ID
        type_name = data["type"]["name"]
        type_id = TYPE_MAP.get(type_name, 17)
        
        moves_dict[move_id] = {
            "name": data["name"].replace("-", " ").title(),
            "type_str": type_name.capitalize(),
            "type_id": type_id,
            "power": power,
            "accuracy": accuracy,
            "pp": data["pp"]
        }
        
    # Polite delay to prevent API rate-limiting
    time.sleep(0.1)
    
    if move_id % 50 == 0:
        print(f"Fetched {move_id} / {MAX_GEN3_MOVE_ID} moves...")

# Save the dictionary locally
with open("emerald_moves.json", "w") as f:
    json.dump(moves_dict, f, indent=4)

print("Successfully saved all 354 moves to emerald_moves.json!")
