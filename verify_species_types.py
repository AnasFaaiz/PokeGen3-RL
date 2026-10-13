import json
with open("species_types.json") as f:
    data = json.load(f)

for name in ["ABSOL", "LOMBRE", "SCEPTILE", "SWELLOW", "LINOONE", "AGGRON"]:
    print(name, data.get(name))
