#!/usr/bin/env python3
"""Live check of BOXMON / PARTYMON / SWAP, with training stopped (the bridge takes one client).

    python training/swap_test.py                      # dry run: read, compute, print the SWAP line
    python training/swap_test.py --apply              # send it, then verify
    python training/swap_test.py --apply --slot 2 --box 0.5

Stand in the overworld (not in a battle) and take an mGBA save state first.
Afterwards compare the printed level/stats with the in-game party summary.
To undo: run again with --slot <same> --box <the box slot the old mon went to>.
"""
import argparse
import socket
import sys

from exp_tables import GROWTH, box_level, exp_at_level, growth_rate, level_from_exp, species_growth, species_names
from mon_stats import ROM_BASE
from mon_stats import parse_boxmon, compute_stats
import rom_info
from swap_policy import BoxEntry, SwapPolicy

NAMES = species_names()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=0, help="party slot to replace (default: lead)")
    ap.add_argument("--box", help="box slot b.s to bring in (default: best in-band mon)")
    ap.add_argument("--area", type=int, default=20, help="wild level of the area, used to pick a mon")
    ap.add_argument("--apply", action="store_true", help="actually send SWAP (default: dry run)")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--force", action="store_true", help="continue past a level-from-exp mismatch")
    a = ap.parse_args()

    sock = socket.create_connection(("127.0.0.1", a.port), timeout=10)
    f = sock.makefile("r")

    def ask(line):
        sock.sendall((line + "\n").encode())
        return f.readline().rstrip("\n")

    inv = dict(kv.split("=", 1) for kv in ask("INVENTORY")[4:].split(";"))
    levels = [int(x) for x in inv["lv"].split(",") if x]
    print(f"party levels (game): {levels}   map {inv['map']}   trainer={inv['trainer']}")

    # 0. Pull the ROM's own species table (growth rate + base stats) for everything we will use.
    party_mons = [parse_boxmon(ask(f"PARTYMON:{i}")) for i in range(len(levels))]
    ids = [m["species"] for m in party_mons if m]
    before = {sid: growth_rate(sid) for sid in ids}
    missing = rom_info.ensure(ask, ids)
    print(f"ROM species table: {'OK' if not missing else f'{missing} species unavailable (address wrong?)'}")
    for sid in sorted(set(ids)):
        ref = {v: k for k, v in GROWTH.items()}
        print(f"  {NAMES.get(sid, sid):<10} growth ROM={ref.get(growth_rate(sid), '?').replace('GROWTH_', '')}"
              f"  reference={ref.get(species_growth().get(sid), '?').replace('GROWTH_', '')}"
              f"  base stats ROM={ROM_BASE.get(sid, '-')}")

    # 1. Check the parse + exp curve against the game: level from exp must match the party's level.
    ok = True
    for i, lv in enumerate(levels):
        m = party_mons[i]
        if not m:
            print(f"  slot {i}: PARTYMON failed"); ok = False; continue
        calc = box_level(m["species"], m["exp"])
        flag = "ok" if calc == lv else "MISMATCH"
        ok &= calc == lv
        print(f"  slot {i}: {NAMES.get(m['species'], m['species'])} game L{lv}, from exp L{calc}  {flag}")
        if calc != lv:
            # Diagnostic: the raw exp, the exp the game's level should need, and which curves give the game's level.
            rate = growth_rate(m["species"])
            fits = [name.replace("GROWTH_", "") for name, r in GROWTH.items() if level_from_exp(r, m["exp"]) == lv]
            print(f"          pid={m['pid']} exp={m['exp']} (assumed {[k for k, v in GROWTH.items() if v == rate][0].replace('GROWTH_', '')}: "
                  f"L{lv} needs {exp_at_level(rate, lv)}..{exp_at_level(rate, lv + 1) - 1})  curves giving L{lv}: {fits or 'none'}")
    if not ok and not a.force:
        sys.exit("level-from-exp does not match the game: fix parsing before swapping anything "
                 "(--force to continue anyway; a wrong level also means wrong stats)")

    # 2. Pick the incoming mon.
    box = ask("BOX")
    entries = []
    parsed = [ent.split(":") for ent in filter(None, box[4:].split(","))]
    rom_info.ensure(ask, [int(p[1]) for p in parsed])
    for pos, sp, exp in parsed:
        entries.append(BoxEntry(pos, int(sp), box_level(int(sp), int(exp))))
    if a.box:
        pick = next((e for e in entries if e.key == a.box), None)
    else:
        cands = SwapPolicy().candidates(a.area, entries)
        pick = max(cands, key=lambda e: e.level) if cands else None
    if not pick:
        sys.exit("no box mon to bring in (use --box b.s or --area)")
    mon = parse_boxmon(ask(f"BOXMON:{pick.key}"))
    rom_info.ensure(ask, [mon["species"]])
    stats = compute_stats(mon)
    print(f"\nincoming: box {pick.key} {NAMES.get(pick.species, pick.species)} L{pick.level}")
    print(f"computed: level={stats[0]} hp={stats[1]} atk={stats[2]} def={stats[3]} spe={stats[4]} spa={stats[5]} spd={stats[6]}")
    line = f"SWAP:{a.slot}:{pick.key}:" + ":".join(map(str, stats))
    print(f"command : {line}")
    if not a.apply:
        print("\ndry run only. Re-run with --apply to send it.")
        return

    # 3. Send it and verify.
    print("\nreply   :", ask(line))
    inv = dict(kv.split("=", 1) for kv in ask("INVENTORY")[4:].split(";"))
    print("party levels now:", inv["lv"])
    new = parse_boxmon(ask(f"PARTYMON:{a.slot}"))
    old = parse_boxmon(ask(f"BOXMON:{pick.key}"))
    print(f"slot {a.slot} now: {NAMES.get(new['species'], new['species'])}   box {pick.key} now: {NAMES.get(old['species'], old['species'])}")
    assert new["species"] == pick.species, "incoming species is not in the party slot"
    print("\nNow open the party summary in the game and compare level, HP, stats and moves with the computed line above.")


if __name__ == "__main__":
    main()
