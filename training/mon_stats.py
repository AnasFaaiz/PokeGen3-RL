"""Gen 3 stat calculation, for building the party tail of a Pokemon swapped in from the PC.

Inputs are what the bridge's BOXMON reply carries: species, exp, EVs and the IV word.
The nature comes from the personality value (pid % 25).
"""
import json

from exp_tables import EMERALD, box_level

# species id -> (hp, atk, def, spe, spa, spd) base stats read from the running ROM (see rom_info.py)
ROM_BASE = {}

_NAMES = {int(k): v for k, v in json.loads((EMERALD / "game_data" / "species_id_to_name.json").read_text()).items()}
_BASE = json.loads((EMERALD / "game_data" / "species_data.json").read_text())


def parse_boxmon(line):
    """BOXMON:<pid>:<species>:<exp>:<6 EVs>:<iv word> -> dict, or None on error."""
    if not line.startswith("BOXMON:") or line.startswith("BOXMON:ERR"):
        return None
    _, pid, species, exp, evs, iv = line.split(":")
    return {"pid": int(pid), "species": int(species), "exp": int(exp),
            "evs": [int(x) for x in evs.split(",")], "iv": int(iv)}


def compute_stats(mon):
    """-> (level, hp, atk, def, spe, spa, spd), or None if the species has no base stats."""
    if mon["species"] in ROM_BASE:
        b = list(ROM_BASE[mon["species"]])
    else:
        base = _BASE.get(_NAMES.get(mon["species"], ""))
        if not base:
            return None
        b = [base["base_hp"], base["base_atk"], base["base_def"], base["base_spd"],
             base["base_spa"], base["base_spdef"]]     # hp, atk, def, spe, spa, spd
    level = box_level(mon["species"], mon["exp"])
    iv = [(mon["iv"] >> (5 * i)) & 31 for i in range(6)]        # hp, atk, def, spe, spa, spd
    ev = mon["evs"]
    hp = (2 * b[0] + iv[0] + ev[0] // 4) * level // 100 + level + 10
    nature = mon["pid"] % 25
    up, down = nature // 5, nature % 5                          # index into atk, def, spe, spa, spd
    stats = []
    for i in range(1, 6):
        s = (2 * b[i] + iv[i] + ev[i] // 4) * level // 100 + 5
        if up != down:
            if i - 1 == up:
                s = s * 110 // 100
            elif i - 1 == down:
                s = s * 90 // 100
        stats.append(s)
    return (level, hp, *stats)


if __name__ == "__main__":
    # Bulbasaur L50, 31 IVs, no EVs, neutral nature (pid % 25 == 0): HP 120, Atk 69, Def 69, Spe 65, SpA 85, SpD 85
    iv = sum(31 << (5 * i) for i in range(6))
    mon = {"pid": 25, "species": 1, "exp": 125000, "evs": [0] * 6, "iv": iv}
    assert compute_stats(mon) == (50, 120, 69, 69, 65, 85, 85), compute_stats(mon)
    mon["pid"] = 1   # nature 1 = Lonely: +Atk -Def
    assert compute_stats(mon)[2:4] == (75, 62)
    print("ok")
