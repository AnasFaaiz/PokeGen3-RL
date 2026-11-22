"""Gen 3 damage and stat estimates for observation features. Pure logic: no sockets, no env import.

The agent used to see only move power, type and effectiveness. Real play also needs "will this
move KO?" and "who moves first?", which come from stats and the damage formula, so they are
computed here. Everything is an estimate: stats come from the ROM base stats, the mon's level and
neutral IV/EV (the game does not expose another mon's real stats in GET_STATE).

Gen 3 specifics that matter here:
  * the physical/special split is by the move's TYPE, not per move (Ghost is physical, Dragon special);
  * Seismic Toss / Night Shade deal the user's level, Sonic Boom 20, Dragon Rage 40,
    Super Fang half the target's HP, and the OHKO moves (accuracy 30) KO outright.
"""
from functools import lru_cache

from mon_stats import ROM_BASE, _BASE, _NAMES

# Game type ids (Normal 0 ... Steel 8 are physical; ??? 9, Fire 10 ... Dark 17 are special).
PHYSICAL_TYPES = frozenset(range(0, 9))
FIXED_DAMAGE = {49: 20, 82: 40}            # Sonic Boom, Dragon Rage
LEVEL_DAMAGE = frozenset({69, 101})        # Seismic Toss, Night Shade
OHKO_MOVES = frozenset({12, 32, 90, 329})  # Guillotine, Horn Drill, Fissure, Sheer Cold
HALF_HP_MOVES = frozenset({162})           # Super Fang
ABILITY_WONDER_GUARD, ABILITY_LEVITATE = 25, 26
TYPE_GROUND = 4
RANDOM_AVG = 0.925                         # damage roll is 85-100%, mean ~92.5%


def base_stats(species):
    """(hp, atk, def, spe, spa, spd) base stats, or None for an unknown species."""
    if species in ROM_BASE:
        return tuple(ROM_BASE[species])
    d = _BASE.get(_NAMES.get(species, ""))
    if not d:
        return None
    return (d["base_hp"], d["base_atk"], d["base_def"], d["base_spd"], d["base_spa"], d["base_spdef"])


def clear_caches():
    """Drop cached stats; rom_info.ensure calls this when ROM base stats arrive for new species."""
    estimate_stats.cache_clear()


@lru_cache(maxsize=4096)
def estimate_stats(species, level):
    """(hp, atk, def, spe, spa, spd) at `level` with neutral IV 15 / no EVs, or None."""
    b = base_stats(species)
    if b is None or level <= 0:
        return None
    hp = (2 * b[0] + 15) * level // 100 + level + 10
    return (hp,) + tuple((2 * x + 15) * level // 100 + 5 for x in b[1:])


def stage_mult(stage):
    """Stat stage -6..+6 -> multiplier."""
    stage = max(-6, min(6, stage))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def expected_damage(move, power, move_type, level, atk, defn, atk_stage, def_stage,
                    stab, eff, target_hp, target_max_hp, def_ability=0):
    """Average HP damage of `move`, or 0 for a status move or an immune target.
    `atk`/`defn` are the already-chosen attack and defence stats (physical or special)."""
    if eff == 0:
        return 0.0
    if def_ability == ABILITY_LEVITATE and move_type == TYPE_GROUND:
        return 0.0
    if def_ability == ABILITY_WONDER_GUARD and eff <= 1:
        return 0.0
    if move in OHKO_MOVES:
        return float(target_hp)
    if move in LEVEL_DAMAGE:
        return float(level)
    if move in FIXED_DAMAGE:
        return float(FIXED_DAMAGE[move])
    if move in HALF_HP_MOVES:
        return max(1.0, target_hp / 2)
    if power <= 0:
        return 0.0
    a = max(1.0, atk * stage_mult(atk_stage))
    d = max(1.0, defn * stage_mult(def_stage))
    base = ((2 * level // 5 + 2) * power * a / d) / 50 + 2
    return base * (1.5 if stab else 1.0) * eff * RANDOM_AVG


def pick_stats(stats, move_type):
    """(attack, defence) from an estimate tuple, chosen by the move's type (Gen 3 split)."""
    _, atk, dfn, _, spa, spd = stats
    return (atk, dfn) if move_type in PHYSICAL_TYPES else (spa, spd)


if __name__ == "__main__":
    # Level 50, atk 100 vs def 100, power 80, STAB, neutral: ((22*80*100/100)/50 + 2) * 1.5 * 0.925
    got = expected_damage(1, 80, 0, 50, 100, 100, 0, 0, True, 1.0, 100, 100)
    assert abs(got - ((22 * 80 / 50 + 2) * 1.5 * 0.925)) < 1e-6, got
    assert expected_damage(1, 80, 0, 50, 100, 100, 0, 0, False, 0.0, 100, 100) == 0.0          # immune
    assert expected_damage(1, 80, 0, 50, 100, 100, 1, 0, False, 1.0, 100, 100) > \
        expected_damage(1, 80, 0, 50, 100, 100, 0, 0, False, 1.0, 100, 100)                   # +1 attack helps
    assert expected_damage(69, 0, 1, 27, 1, 999, 0, 0, False, 1.0, 100, 100) == 27.0            # Seismic Toss = level
    assert expected_damage(82, 0, 16, 5, 1, 999, 0, 0, False, 1.0, 100, 100) == 40.0            # Dragon Rage
    assert expected_damage(90, 0, 4, 5, 1, 1, 0, 0, False, 1.0, 77, 100) == 77.0                # OHKO = target HP
    assert expected_damage(90, 0, 4, 5, 1, 1, 0, 0, False, 0.0, 77, 100) == 0.0                 # ...unless immune
    assert expected_damage(162, 0, 0, 5, 1, 1, 0, 0, False, 1.0, 50, 100) == 25.0               # Super Fang
    assert expected_damage(1, 0, 0, 50, 100, 100, 0, 0, True, 1.0, 100, 100) == 0.0             # status move
    assert expected_damage(1, 80, 4, 50, 100, 100, 0, 0, False, 1.0, 100, 100, 26) == 0.0       # Levitate
    assert expected_damage(1, 80, 0, 50, 100, 100, 0, 0, False, 1.0, 100, 100, 25) == 0.0       # Wonder Guard, neutral
    assert expected_damage(1, 80, 0, 50, 100, 100, 0, 0, False, 2.0, 100, 100, 25) > 0          # ...super effective hits
    assert stage_mult(0) == 1 and stage_mult(2) == 2 and stage_mult(-2) == 0.5 and stage_mult(9) == stage_mult(6)
    s = estimate_stats(1, 50)                       # Bulbasaur L50, IV 15, no EV, neutral nature
    assert s == (112, 61, 61, 57, 77, 77), s
    assert estimate_stats(1, 0) is None and estimate_stats(0, 50) is None
    assert pick_stats((1, 2, 3, 4, 5, 6), 0) == (2, 3) and pick_stats((1, 2, 3, 4, 5, 6), 10) == (5, 6)
    print("ok")
