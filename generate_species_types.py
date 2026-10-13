import re
import json

with open("species.h") as f:
    content = f.read()

# matches: #define SPECIES_ABSOL 376
pattern = re.compile(r"#define\s+SPECIES_(\w+)\s+(\d+)")

id_to_name = {}
for match in pattern.finditer(content):
    name, species_id = match.groups()
    id_to_name[int(species_id)] = name

with open("species_id_to_name.json", "w") as f:
    json.dump(id_to_name, f, indent=2)

print(f"Extracted {len(id_to_name)} species IDs")
