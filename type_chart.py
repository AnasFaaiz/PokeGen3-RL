import requests
import json
import time
import certifi

# Gen 3 has 17 usable types (no Fairy yet) + "unknown"/"???" as a placeholder = 18
TYPE_MAP = {
    "normal": 0, "fighting": 1, "flying": 2, "poison": 3, "ground": 4,
    "rock": 5, "bug": 6, "ghost": 7, "steel": 8, "fire": 9, "water": 10,
    "grass": 11, "electric": 12, "psychic": 13, "ice": 14, "dragon": 15,
    "dark": 16, "unknown": 17
}

# Reverse map for API calls: name -> id, and we fetch by name since
# PokeAPI's type IDs don't perfectly align with Gen 3's internal ordering
TYPE_NAMES = list(TYPE_MAP.keys())

type_chart = {}  # type_chart[attacking_id][defending_id] = multiplier

print("Fetching Gen 3 Type Effectiveness Chart from PokeAPI...")

for type_name in TYPE_NAMES:
    if type_name == "unknown":
        continue  # no real damage relations for this placeholder type

    url = f"https://pokeapi.co/api/v2/type/{type_name}/"
    response = requests.get(url, verify=certifi.where())

    if response.status_code != 200:
        print(f"Failed to fetch {type_name}, skipping")
        continue

    data = response.json()
    attacking_id = TYPE_MAP[type_name]
    type_chart[attacking_id] = {}

    relations = data["damage_relations"]

    for entry in relations["double_damage_to"]:
        defending_id = TYPE_MAP.get(entry["name"])
        if defending_id is not None:
            type_chart[attacking_id][defending_id] = 2.0

    for entry in relations["half_damage_to"]:
        defending_id = TYPE_MAP.get(entry["name"])
        if defending_id is not None:
            type_chart[attacking_id][defending_id] = 0.5

    for entry in relations["no_damage_to"]:
        defending_id = TYPE_MAP.get(entry["name"])
        if defending_id is not None:
            type_chart[attacking_id][defending_id] = 0.0

    print(f"Fetched {type_name} ({attacking_id})")
    time.sleep(0.1)

with open("type_chart.json", "w") as f:
    json.dump(type_chart, f, indent=4)

print("Successfully saved type effectiveness chart to type_chart.json!")
