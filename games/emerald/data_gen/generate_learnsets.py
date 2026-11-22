"""Build game_data/learnsets.json (level-up learnsets by internal species id) from the pret/pokeemerald decomp.

Inputs (public decomp files, kept under reference/learnsets/):
  src_data_pokemon_level_up_learnsets.h          sXxxLevelUpLearnset[] = { LEVEL_UP_MOVE(level, MOVE_X), ... }
  src_data_pokemon_level_up_learnset_pointers.h  gLevelUpLearnsets[SPECIES_X] = sXxxLevelUpLearnset
  include_constants_moves.h                      MOVE_X -> id
and reference/species.h for SPECIES_X -> internal species id.

    python generate_learnsets.py

Output: {"<species id>": [[level, move id], ...]} sorted by level. Counts of anything that did not resolve are printed.
"""
import json
import re
from pathlib import Path

EMERALD = Path(__file__).resolve().parent.parent
REF = EMERALD / "reference"
OUT = EMERALD / "game_data" / "learnsets.json"


def defines(path):
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"^#define\s+(\w+)\s+(\d+)\b", Path(path).read_text(), re.M)}


def parse_learnsets(text, moves):
    """symbol -> [[level, move id], ...]; also returns the move names that were not in the constants."""
    out, unknown = {}, []
    for m in re.finditer(r"static const u16 (\w+)\[\]\s*=\s*\{(.*?)\};", text, re.S):
        entries = []
        for lvl, mv in re.findall(r"LEVEL_UP_MOVE\(\s*(\d+)\s*,\s*(\w+)\s*\)", m.group(2)):
            if mv in moves:
                entries.append([int(lvl), moves[mv]])
            else:
                unknown.append(mv)
        out[m.group(1)] = sorted(entries, key=lambda e: (e[0], e[1]))
    return out, unknown


def main():
    moves = defines(REF / "learnsets" / "include_constants_moves.h")
    species = defines(REF / "species.h")
    by_symbol, unknown = parse_learnsets((REF / "learnsets" / "src_data_pokemon_level_up_learnsets.h").read_text(), moves)
    result, no_symbol = {}, []
    pointers = (REF / "learnsets" / "src_data_pokemon_level_up_learnset_pointers.h").read_text()
    for sp, sym in re.findall(r"\[(SPECIES_\w+)\]\s*=\s*(\w+)", pointers):
        sid = species.get(sp)
        if not sid or sp == "SPECIES_NONE":
            continue
        if sym not in by_symbol:
            no_symbol.append(sp)
            continue
        result[str(sid)] = by_symbol[sym]
    OUT.write_text(json.dumps(result, separators=(",", ":")))
    print(f"species with a learnset: {len(result)}; learnset tables parsed: {len(by_symbol)}; "
          f"moves in constants: {len(moves)}")
    if unknown:
        print(f"WARNING {len(set(unknown))} move names not in moves.h: {sorted(set(unknown))[:8]}")
    if no_symbol:
        print(f"WARNING {len(no_symbol)} species point at a missing table: {no_symbol[:8]}")


if __name__ == "__main__":
    main()
