"""Which moves a Pokemon should know. Pure logic: no sockets, no game access.

Between battles the agent may swap one of a mon's moves for a level-up move it has not learned (the game itself only
asks while the mon is in a fight, and the bridge declines that prompt). This module scores moves from the mon's own
strengths and proposes the replacements worth making:

  * every move is valued as the average damage it deals to a reference enemy of the same level, as a percent of that
    enemy's HP, times accuracy and a "real-fight" factor for awkward moves (two-turn, recharge, self-KO...);
    damage_calc supplies the Gen 3 formula, so STAB, fixed-damage moves and the physical/special split (by move TYPE
    in Gen 3) fall out of the mon's own Attack / Sp. Atk stats;
  * non-damaging moves get a flat value by role (heal / status / setup);
  * a damaging move that hits types the rest of the set does not hit gets a coverage bonus;
  * a replacement must keep at least two damaging moves (when the mon has them), never overwrite an HM move and never
    teach one (an HM move cannot be forgotten), and must clear a minimum gain so the set does not churn.

`plan_teach` returns the candidates ranked by gain. A scripted rule or a learned policy can choose among them.
"""
import json
from functools import lru_cache
from pathlib import Path

import damage_calc as dc
import move_tags
from mon_stats import _BASE, _NAMES

_DATA = Path(__file__).resolve().parent.parent / "games" / "emerald" / "game_data"

MIN_DELTA = 6.0                  # smallest set-value gain worth a swap (percent-of-enemy-HP points)
ROLE_VALUE = {"heal": 18.0, "status": 16.0, "setup": 14.0, "other": 2.0}
COVERAGE_PER_TYPE = 1.2          # per newly super-effective defending type
COVERAGE_CAP = 10.0
MAX_UTILITY_MOVES = 2            # at most this many non-damaging moves in a set of four
MIN_DAMAGING = 2                 # keep at least this many damaging moves (if the mon has them)
REFERENCE_BASE = 70              # a typical enemy: every base stat 70
TYPE_IDS = {"NORMAL": 0, "FIGHTING": 1, "FLYING": 2, "POISON": 3, "GROUND": 4, "ROCK": 5, "BUG": 6, "GHOST": 7,
            "STEEL": 8, "FIRE": 10, "WATER": 11, "GRASS": 12, "ELECTRIC": 13, "PSYCHIC": 14, "ICE": 15,
            "DRAGON": 16, "DARK": 17}
DEFENDING_TYPES = tuple(TYPE_IDS.values())


@lru_cache(maxsize=1)
def _moves():
    raw = json.loads((_DATA / "emerald_moves.json").read_text())
    out = {}
    for k, v in raw.items():
        t = v["type_id"]
        out[int(k)] = {"name": v["name"], "power": v.get("power") or 0, "acc": v.get("accuracy") or 100,
                       "type": t if t <= 8 else t + 1,          # the game's numbering (see env.py)
                       "pp": v.get("pp") or 5}
    return out


@lru_cache(maxsize=1)
def _learnsets():
    try:
        raw = json.loads((_DATA / "learnsets.json").read_text())
    except (OSError, ValueError):
        return {}
    return {int(k): [(lvl, mv) for lvl, mv in v] for k, v in raw.items()}


def move_info(move):
    return _moves().get(move)


def base_pp(move):
    info = move_info(move)
    return info["pp"] if info else 5


def species_types(species):
    d = _BASE.get(_NAMES.get(species, ""))
    if not d:
        return (0, 0)
    return (TYPE_IDS.get(d["type1"], 0), TYPE_IDS.get(d["type2"], TYPE_IDS.get(d["type1"], 0)))


def is_damaging(move):
    info = move_info(move)
    if not info:
        return False
    return (info["power"] > 0 or move in dc.FIXED_DAMAGE or move in dc.LEVEL_DAMAGE or move in dc.OHKO_MOVES
            or move in dc.HALF_HP_MOVES)


def role(move):
    heal, status, setup = move_tags.tags(move)
    return "heal" if heal else "status" if status else "setup" if setup else "other"


def _eff_default(move_type, d1, d2):
    import env                                    # imported late: env imports this module
    return env.get_effectiveness(move_type, d1, d2)


def _reference_enemy(level):
    hp = (2 * REFERENCE_BASE + 15) * level // 100 + level + 10
    stat = (2 * REFERENCE_BASE + 15) * level // 100 + 5
    return hp, stat


def damage_value(move, mon):
    """Average damage of `move` as a percent of a same-level reference enemy's HP, times accuracy and the quirk factor."""
    info = move_info(move)
    if not info or not is_damaging(move):
        return 0.0
    stats = mon.get("stats") or dc.estimate_stats(mon["species"], mon["level"])
    if not stats:
        return 0.0
    hp, ref = _reference_enemy(mon["level"])
    atk = dc.pick_stats(stats, info["type"])[0]       # the attacking stat for this move's category
    dmg = dc.expected_damage(move, info["power"], info["type"], mon["level"], atk, ref, 0, 0,
                             info["type"] in mon["types"], 1.0, hp, hp)     # the reference enemy defends with `ref`
    return 100.0 * dmg / hp * info["acc"] / 100.0 * move_tags.quirk(move)


def _covered(moves, eff_fn):
    out = set()
    for m in moves:
        info = move_info(m)
        if info and is_damaging(m):
            for d in DEFENDING_TYPES:
                if eff_fn(info["type"], d, d) >= 2.0:
                    out.add(d)
    return out


def set_value(moves, mon, eff_fn=_eff_default):
    """Total value of a moveset (0 for an empty slot)."""
    real = [m for m in moves if m and move_info(m)]
    total = 0.0
    for i, m in enumerate(real):
        if is_damaging(m):
            v = damage_value(m, mon)
            others = [x for j, x in enumerate(real) if j != i]
            new = _covered([m], eff_fn) - _covered(others, eff_fn)
            total += v + (min(COVERAGE_CAP, COVERAGE_PER_TYPE * len(new)) if v > 1.0 else 0.0)
        else:
            total += ROLE_VALUE[role(m)] * move_tags.quirk(m)
    # Utility moves beyond the cap are dead weight: a set of four utility moves cannot win a fight.
    utility = sorted((ROLE_VALUE[role(m)] * move_tags.quirk(m) for m in real if not is_damaging(m)), reverse=True)
    total -= sum(utility[MAX_UTILITY_MOVES:])
    return total


def _valid_set(moves):
    """Rules a resulting set must satisfy (they only bind when the mon had enough damaging moves to begin with)."""
    real = [m for m in moves if m]
    damaging = sum(is_damaging(m) for m in real)
    return damaging >= min(MIN_DAMAGING, len(real))


def learnable(species, level, known):
    """Level-up moves at or below `level` that the mon doesn't know and that are safe to teach."""
    seen, out = set(known), []
    for lvl, mv in _learnsets().get(species, []):
        if lvl <= level and mv not in seen and mv and mv not in move_tags.HM_MOVES and move_info(mv):
            if move_tags.quirk(mv) > 0.30:        # conditional / self-KO / no-effect moves are not worth teaching
                out.append(mv)
                seen.add(mv)
    return out


def best_replacement(known, new_move, mon, eff_fn=_eff_default):
    """(slot, delta) of the replacement that leaves the best set, or None if no slot may be replaced.
    Empty slots are filled first (nothing is lost). HM moves are never overwritten."""
    base = set_value(known, mon, eff_fn)
    best = None
    for slot, old in enumerate(known):
        if old in move_tags.HM_MOVES:
            continue
        trial = list(known)
        trial[slot] = new_move
        if old and not _valid_set(trial) and _valid_set(known):
            continue
        delta = set_value(trial, mon, eff_fn) - base
        if best is None or delta > best[1] + 1e-9:
            best = (slot, delta)
    return best


def plan_teach(mon, known, k=4, min_delta=MIN_DELTA, eff_fn=_eff_default):
    """Candidates worth teaching, best first: [{"move", "slot", "replaced", "delta", "value"}, ...] (at most k).
    `mon` = {"species", "level", "types" (optional), "stats" (optional)}; `known` = four move ids (0 = empty)."""
    mon = dict(mon)
    mon.setdefault("types", species_types(mon["species"]))
    known = (list(known) + [0, 0, 0, 0])[:4]
    out = []
    for mv in learnable(mon["species"], mon["level"], [m for m in known if m]):
        rep = best_replacement(known, mv, mon, eff_fn)
        if rep is None:
            continue
        slot, delta = rep
        if delta >= min_delta:
            out.append({"move": mv, "slot": slot, "replaced": known[slot], "delta": delta,
                        "value": damage_value(mv, mon) if is_damaging(mv) else ROLE_VALUE[role(mv)]})
    out.sort(key=lambda c: (-c["delta"], c["move"]))
    return out[:k]


def apply_teach(known, slot, move):
    new = (list(known) + [0, 0, 0, 0])[:4]
    new[slot] = move
    return new


def parse_monmoves(line):
    """MONMOVES:<pid>:<species>:<level>:<m1>,<m2>,<m3>,<m4>:<pp1>,..,<pp4>:<ppBonuses>:<checksum_ok> -> dict, or None
    (an error reply, or anything that doesn't parse)."""
    if not line.startswith("MONMOVES:") or line.startswith("MONMOVES:ERR"):
        return None
    try:
        _, pid, species, level, moves, pp, bonuses, ok = line.split(":")
        return {"pid": int(pid), "species": int(species), "level": int(level),
                "moves": [int(x) for x in moves.split(",")], "pp": [int(x) for x in pp.split(",")],
                "bonuses": int(bonuses), "ok": ok == "1"}
    except ValueError:
        return None
