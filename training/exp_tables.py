"""Gen 3 experience curves and box-mon level lookup.

PC box mons store no level, only experience. The growth rate per species
comes from games/emerald/reference/species_info.h; the curve formulas are the
standard Gen 3 ones (pret's gExperienceTables, levels 1-100).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

EMERALD = Path(__file__).resolve().parent.parent / "games" / "emerald"

GROWTH = {
    "GROWTH_MEDIUM_FAST": 0, "GROWTH_ERRATIC": 1, "GROWTH_FLUCTUATING": 2,
    "GROWTH_MEDIUM_SLOW": 3, "GROWTH_FAST": 4, "GROWTH_SLOW": 5,
}


def exp_at_level(rate, n):
    if n <= 1:
        return 0
    if rate == 0:
        return n ** 3
    if rate == 4:
        return 4 * n ** 3 // 5
    if rate == 5:
        return 5 * n ** 3 // 4
    if rate == 3:
        return max(0, 6 * n ** 3 // 5 - 15 * n ** 2 + 100 * n - 140)
    if rate == 1:
        if n <= 50:
            return n ** 3 * (100 - n) // 50
        if n <= 68:
            return n ** 3 * (150 - n) // 100
        if n <= 98:
            return n ** 3 * ((1911 - 10 * n) // 3) // 500
        return n ** 3 * (160 - n) // 100
    if rate == 2:
        if n <= 15:
            return n ** 3 * ((n + 1) // 3 + 24) // 50
        if n <= 35:
            return n ** 3 * (n + 14) // 50
        return n ** 3 * (n // 2 + 32) // 50
    raise ValueError(rate)


def level_from_exp(rate, exp):
    lvl = 1
    for n in range(2, 101):
        if exp >= exp_at_level(rate, n):
            lvl = n
        else:
            break
    return lvl


@lru_cache(maxsize=1)
def species_growth():
    """species id -> growth rate id."""
    ref = EMERALD / "reference"
    ids = {m.group(1): int(m.group(2))
           for m in re.finditer(r"#define SPECIES_(\w+)\s+(\d+)", (ref / "species.h").read_text())}
    out, cur = {}, None
    for line in (ref / "species_info.h").read_text().splitlines():
        m = re.search(r"\[SPECIES_(\w+)\]\s*=", line)
        if m:
            cur = m.group(1)
        g = re.search(r"\.growthRate\s*=\s*(GROWTH_\w+)", line)
        if g and cur in ids:
            out[ids[cur]] = GROWTH[g.group(1)]
            cur = None
    return out


@lru_cache(maxsize=1)
def species_names():
    d = json.loads((EMERALD / "game_data" / "species_id_to_name.json").read_text())
    return {int(k): v for k, v in d.items()}


def box_level(species, exp):
    return level_from_exp(species_growth().get(species, 0), exp)


if __name__ == "__main__":
    # Sanity: known Gen 3 totals at level 100.
    assert exp_at_level(0, 100) == 1_000_000
    assert exp_at_level(4, 100) == 800_000
    assert exp_at_level(5, 100) == 1_250_000
    assert exp_at_level(3, 100) == 1_059_860
    assert exp_at_level(1, 100) == 600_000
    assert exp_at_level(2, 100) == 1_640_000
    print("species with growth data:", len(species_growth()))
    print("ok")
