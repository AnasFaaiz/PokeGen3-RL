"""Build trainers.json and trainer_objects.json from the pret/pokeemerald decomp.

Inputs (public decomp files, see games/emerald/reference/trainers/ for the trainer tables):
  reference/trainers/constants_opponents.h   TRAINER_* name -> id
  reference/trainers/src_trainers.h          gTrainers[]: class, name, double battle, party symbol
  reference/trainers/src_trainer_parties.h   sParty_*[]: species, level, moves, held item
  <decomp>/data/maps/*/map.json              object events (position, trainer type, sight, script)
  <decomp>/data/maps/*/scripts.inc           which TRAINER_* each trainer script starts a fight with
  <decomp>/data/maps/map_groups.json         map group / number, which is what the bridge reports as "map=g.n"

    python generate_trainers.py <path to a pokeemerald checkout containing data/maps>

Outputs (games/emerald/game_data/):
  trainers.json         {"<id>": {"const", "name", "class", "double", "party": [{"species", "level", "moves", "item"}]}}
  trainer_objects.json  {"<group>.<num>": [{"id", "x", "y", "sight", "script", "type", "flag"}]}
Parse failures are counted and printed, never silently skipped.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMERALD = HERE.parent
REF = EMERALD / "reference" / "trainers"
OUT = EMERALD / "game_data"


def read_defines(path):
    out = {}
    for m in re.finditer(r"^#define\s+(\w+)\s+(\d+)\s*(?://.*)?$", Path(path).read_text(), re.M):
        out[m.group(1)] = int(m.group(2))
    return out


def parse_parties(text):
    """sParty_Name -> [{"species", "level", "moves", "item"}]"""
    parties = {}
    for m in re.finditer(r"sParty_(\w+)\[\]\s*=\s*\{(.*?)\n\};", text, re.S):
        mons = []
        for body in re.findall(r"\{(.*?)\n\s*\}", m.group(2), re.S):
            lvl = re.search(r"\.lvl\s*=\s*(\d+)", body)
            sp = re.search(r"\.species\s*=\s*(\w+)", body)
            if not (lvl and sp):
                continue
            mv = re.search(r"\.moves\s*=\s*\{([^}]*)\}", body)
            it = re.search(r"\.heldItem\s*=\s*(\w+)", body)
            moves = [x.strip() for x in mv.group(1).split(",")] if mv else []
            mons.append({
                "species": sp.group(1),
                "level": int(lvl.group(1)),
                "moves": [x for x in moves if x and x != "MOVE_NONE"],
                "item": it.group(1) if it and it.group(1) != "ITEM_NONE" else None,
            })
        parties[m.group(1)] = mons
    return parties


def parse_trainers(text, ids, parties):
    trainers, bad = {}, []
    for m in re.finditer(r"\[(TRAINER_\w+)\]\s*=\s*\{(.*?)\n    \},", text, re.S):
        const, body = m.group(1), m.group(2)
        if const not in ids:
            bad.append(const)
            continue
        name = re.search(r'\.trainerName\s*=\s*_\("([^"]*)"\)', body)
        cls = re.search(r"\.trainerClass\s*=\s*(\w+)", body)
        dbl = re.search(r"\.doubleBattle\s*=\s*(\w+)", body)
        party = re.search(r"\.party\s*=\s*\w+\((\w+)\)", body)
        sym = party.group(1).replace("sParty_", "", 1) if party else None
        if sym is not None and sym not in parties:
            bad.append(f"{const}:{sym}")
        trainers[str(ids[const])] = {
            "const": const,
            "name": name.group(1) if name else "",
            "class": cls.group(1) if cls else "",
            "double": bool(dbl and dbl.group(1) == "TRUE"),
            "party": parties.get(sym, []),
        }
    return trainers, bad


def parse_maps(decomp):
    maps_dir = Path(decomp) / "data" / "maps"
    groups = json.loads((maps_dir / "map_groups.json").read_text())
    keys = {}                                           # map folder name -> "group.num"
    for g, name in enumerate(groups["group_order"]):
        for n, folder in enumerate(groups[name]):
            keys[folder] = f"{g}.{n}"
    objects, unresolved = {}, []
    for folder, key in keys.items():
        mp = maps_dir / folder / "map.json"
        if not mp.exists():
            continue
        scripts = (maps_dir / folder / "scripts.inc")
        label_trainer = {}
        if scripts.exists():
            label = None                                # the label whose body we are in
            for line in scripts.read_text().splitlines():
                m = re.match(r"^(\w+)::", line)
                if m:
                    label = m.group(1)
                    continue
                m = re.match(r"\s*trainerbattle\w*\s+(?:TRAINER_BATTLE_\w+\s*,\s*)?(TRAINER_\w+)", line)
                if m and label:
                    label_trainer.setdefault(label, m.group(1))
        for ev in json.loads(mp.read_text()).get("object_events", []):
            if ev.get("trainer_type", "TRAINER_TYPE_NONE") == "TRAINER_TYPE_NONE":
                continue
            const = label_trainer.get(ev.get("script"))
            if const is None:
                unresolved.append(f"{folder}:{ev.get('script')}")
            objects.setdefault(key, []).append({
                "const": const, "x": ev["x"], "y": ev["y"],
                "sight": int(ev.get("trainer_sight_or_berry_tree_id", 0) or 0),
                "script": ev.get("script"), "type": ev["trainer_type"], "flag": ev.get("flag"),
                "map": folder,
            })
    return objects, unresolved


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ids = read_defines(REF / "constants_opponents.h")
    parties = parse_parties((REF / "src_trainer_parties.h").read_text())
    trainers, bad = parse_trainers((REF / "src_trainers.h").read_text(), ids, parties)
    objects, unresolved = parse_maps(sys.argv[1])
    for objs in objects.values():                       # replace the constant with the numeric id
        for o in objs:
            o["id"] = ids.get(o.pop("const"), None) if o["const"] else None
    (OUT / "trainers.json").write_text(json.dumps(trainers, indent=1))
    (OUT / "trainer_objects.json").write_text(json.dumps(objects, indent=1))
    n_obj = sum(len(v) for v in objects.values())
    print(f"trainers: {len(trainers)} ({sum(1 for t in trainers.values() if t['party'])} with a party); "
          f"parties parsed: {len(parties)}")
    print(f"trainer objects: {n_obj} on {len(objects)} maps; unresolved scripts: {len(unresolved)}")
    if bad:
        print(f"WARNING {len(bad)} trainer entries with a missing id/party: {bad[:8]}")
    if unresolved:
        print(f"WARNING unresolved trainer scripts (no trainerbattle found): {unresolved[:8]}")


if __name__ == "__main__":
    main()
