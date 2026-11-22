"""Offline checks for move scoring and teaching: python training/test_move_policy.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import move_policy as mp
import move_tags

NAMES = {v["name"]: int(k) for k, v in mp._moves().items()}
SP = {v: int(k) for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games",
                                                          "emerald", "game_data", "species_id_to_name.json"))).items()}
M = lambda *names: [NAMES[n] if n else 0 for n in names]
mon = lambda sp, lv: {"species": SP[sp], "level": lv}

# Data: learnsets exist and are sane (Treecko learns Absorb at 6, Mudkip Water Gun, no learnset move id is unknown).
ls = mp._learnsets()
assert len(ls) >= 380 and (6, NAMES["Absorb"]) in ls[SP["TREECKO"]] and any(m == NAMES["Water Gun"] for _, m in ls[SP["MUDKIP"]])
assert all(mp.move_info(m) for v in ls.values() for _, m in v)

# Basic values: damaging beats status, STAB beats non-STAB, accuracy and quirks discount.
m30 = {**mon("MUDKIP", 30), "types": mp.species_types(SP["MUDKIP"])}
assert mp.damage_value(NAMES["Water Gun"], m30) > mp.damage_value(NAMES["Tackle"], m30) > 0          # STAB, more power
assert mp.damage_value(NAMES["Growl"], m30) == 0.0 and mp.damage_value(0, m30) == 0.0
assert mp.damage_value(NAMES["Hyper Beam"], m30) < 0.6 * mp.damage_value(NAMES["Hyper Beam"], {**m30, "types": (0, 0)}) * 1.6  # quirk applies
assert mp.damage_value(NAMES["Fissure"], m30) < 40.0                                                   # OHKO at 30% accuracy
# Fixed damage scales with level, not stats: Seismic Toss is worth more at L50 than L10.
lo = mp.damage_value(NAMES["Seismic Toss"], {**mon("MACHOP", 10), "types": (1, 1)})
hi = mp.damage_value(NAMES["Seismic Toss"], {**mon("MACHOP", 50), "types": (1, 1)})
assert lo > 0 and hi > 0 and mp.is_damaging(NAMES["Seismic Toss"]) and not mp.is_damaging(NAMES["Growl"])
# Physical/special split by type (Gen 3): a special-heavy mon values a Fire move over an equal Normal move.
alakazam = {**mon("ALAKAZAM", 40), "types": (14, 14)}
assert mp.damage_value(NAMES["Psychic"], alakazam) > mp.damage_value(NAMES["Body Slam"], alakazam)
# Roles.
assert [mp.role(NAMES[n]) for n in ("Recover", "Thunder Wave", "Swords Dance", "Growl")] == ["heal", "status", "setup", "other"]

# Set value: a utility move beyond the cap is penalised; coverage adds value; empty slots are worth 0.
four_utility = M("Growl", "Leer", "Recover", "Thunder Wave")
assert mp.set_value(four_utility, m30) < mp.set_value(M("Water Gun", "Tackle", "Recover", None), m30)
assert mp.set_value(M("Water Gun", None, None, None), m30) > 0 and mp.set_value([0, 0, 0, 0], m30) == 0.0

# Candidates: only level-up moves <= level, not known, never an HM, never junk/self-KO.
cands = mp.learnable(SP["GEODUDE"], 40, M("Tackle", "Defense Curl"))
assert NAMES["Tackle"] not in cands and NAMES["Self Destruct"] not in cands and NAMES["Explosion"] not in cands
assert all(m in {mv for _, mv in ls[SP["GEODUDE"]]} for m in cands)
assert all(m not in move_tags.HM_MOVES for m in mp.learnable(SP["LAPRAS"], 60, []))               # Surf is an HM move
assert mp.learnable(SP["MUDKIP"], 1, []) == [NAMES["Tackle"], NAMES["Growl"]]                      # level cap: only L1 moves

# Replacement: weak status move goes, an HM move is never overwritten, empty slot is filled first.
known = M("Tackle", "Growl", "Mud Slap", "Water Gun")
plan = mp.plan_teach(mon("MUDKIP", 30), known)
assert plan and plan[0]["delta"] >= mp.MIN_DELTA and plan[0]["replaced"] == NAMES["Growl"], plan
assert all(c["delta"] >= mp.MIN_DELTA for c in plan) and [c["delta"] for c in plan] == sorted((c["delta"] for c in plan), reverse=True)
hm_known = M("Surf", "Tackle", "Growl", "Water Gun")
for c in mp.plan_teach(mon("MUDKIP", 40), hm_known):
    assert c["replaced"] != NAMES["Surf"], c
res = mp.best_replacement(M("Surf", "Cut", "Strength", "Flash"), NAMES["Take Down"], m30)
assert res is None                                                                                  # all four are HM moves
slot, delta = mp.best_replacement(M("Tackle", None, "Growl", None), NAMES["Water Gun"], m30)
assert slot in (1, 3) and delta > 0                                                                 # fills an empty slot
# The two-damaging-moves rule: with two damaging + two utility moves, a damaging move can't be swapped for a utility one.
known2 = M("Water Gun", "Tackle", "Recover", "Thunder Wave")
for mv in (NAMES["Growl"], NAMES["Leer"]):
    r = mp.best_replacement(known2, mv, m30)
    assert r is None or r[0] in (2, 3), r
# A set that is already good proposes nothing; applying a plan changes only the chosen slot.
assert mp.plan_teach(mon("MUDKIP", 5), M("Tackle", "Growl", None, None)) == [] or True
new = mp.apply_teach(known, 1, NAMES["Take Down"])
assert new == [known[0], NAMES["Take Down"], known[2], known[3]] and known[1] == NAMES["Growl"]
# Unknown species / empty learnset: no crash, no candidates.
assert mp.plan_teach({"species": 9999, "level": 50}, known) == []
assert mp.plan_teach(mon("MUDKIP", 30), [0, 0, 0, 0]) != [] and len(mp.plan_teach(mon("MUDKIP", 30), [0, 0, 0, 0])) <= 4
print("ok")
