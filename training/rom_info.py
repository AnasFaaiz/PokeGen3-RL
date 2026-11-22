"""Per-species growth rate and base stats, read from the running ROM through the bridge.

The reference files in games/emerald/reference are not guaranteed to match the ROM being
played (a live check showed different growth curves for several species), so level-from-exp
and stats use the ROM's own table whenever the bridge can supply it. If it cannot (the
SPECIES command errors, or the table address is wrong), the reference files are used and
`ensure` says so once.
"""
import exp_tables
import mon_stats

_warned = False


def parse_species(line):
    """SPECIES:<id>:<growth>:<hp>:<atk>:<def>:<spe>:<spa>:<spd>:<t1>:<t2> -> dict, or None."""
    if not line.startswith("SPECIES:") or line.startswith("SPECIES:ERR"):
        return None
    f = [int(x) for x in line.split(":")[1:]]
    return {"species": f[0], "growth": f[1], "base": tuple(f[2:8]), "types": tuple(f[8:10])}


def ensure(request, species_ids):
    """Fetch and cache ROM data for any species not seen yet. `request` sends one command and
    returns the reply line. Returns the number of species the ROM could not supply."""
    global _warned
    missing = 0
    changed = False
    for sid in set(species_ids):
        if sid in exp_tables.ROM_GROWTH or sid <= 0:
            continue
        info = parse_species(request(f"SPECIES:{sid}"))
        if not info:
            missing += 1
            continue
        exp_tables.ROM_GROWTH[sid] = info["growth"]
        mon_stats.ROM_BASE[sid] = info["base"]
        changed = True
    if changed:
        import damage_calc          # cached stat estimates may have used the reference table
        damage_calc.clear_caches()
    if missing and not _warned:
        _warned = True
        print(f"[rom] SPECIES unavailable for {missing} species; using reference tables "
              "(levels/stats may be off for this ROM)")
    return missing


if __name__ == "__main__":
    info = parse_species("SPECIES:292:4:50:70:50:65:90:50:6:2")
    assert info["growth"] == 4 and info["base"] == (50, 70, 50, 65, 90, 50)
    assert parse_species("SPECIES:ERR:bad_species") is None
    calls = []
    def fake(cmd):
        calls.append(cmd)
        return "SPECIES:292:4:50:70:50:65:90:50:6:2" if cmd == "SPECIES:292" else "SPECIES:ERR:x"
    assert ensure(fake, [292, 292, 0, 5]) == 1 and calls == ["SPECIES:292", "SPECIES:5"] or sorted(calls) == ["SPECIES:292", "SPECIES:5"]
    assert exp_tables.growth_rate(292) == 4 and mon_stats.ROM_BASE[292][0] == 50
    # Beautifly exp 7320 is L20 on the Fast curve, which is what the live game reported.
    assert exp_tables.box_level(292, 7320) == 20
    print("ok")
