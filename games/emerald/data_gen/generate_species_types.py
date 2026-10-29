import re
import json

with open("species_info.h") as f:
    content = f.read()

pattern = re.compile(
    r"\[SPECIES_(\w+)\]\s*=\s*\{"
    r"(?:(?!\[SPECIES_).)*?"  # don't let .*? cross into the next [SPECIES_...] block
    r"\.baseHP\s*=\s*(\d+),"
    r"\s*\.baseAttack\s*=\s*(\d+),"
    r"\s*\.baseDefense\s*=\s*(\d+),"
    r"\s*\.baseSpeed\s*=\s*(\d+),"
    r"\s*\.baseSpAttack\s*=\s*(\d+),"
    r"\s*\.baseSpDefense\s*=\s*(\d+),"
    r"\s*\.types\s*=\s*\{\s*TYPE_(\w+)\s*,\s*TYPE_(\w+)\s*\}",
    re.DOTALL
)

species_data = {}
for match in pattern.finditer(content):
    name, hp, atk, dfn, spd, spa, spdef, type1, type2 = match.groups()
    total = int(hp) + int(atk) + int(dfn) + int(spd) + int(spa) + int(spdef)
    species_data[name] = {
        "type1": type1,
        "type2": type2,
        "base_hp": int(hp),
        "base_atk": int(atk),
        "base_def": int(dfn),
        "base_spd": int(spd),
        "base_spa": int(spa),
        "base_spdef": int(spdef),
        "base_total": total,
    }

with open("species_data.json", "w") as f:
    json.dump(species_data, f, indent=2)

print(f"Extracted {len(species_data)} species with full base stats")
