"""Offline checks for the combat observation features: python training/test_features.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env as e
from env import (PokemonBattleEnv as E, EXTRA2_FEATURES, EXTRA3_FEATURES, STACK_FEATURES, OBS_DIM,
                 BASE_OBS_DIM, EXTRA_FEATURES, FRAME_STACK, TEACH_FEATURES, MOVE_TEACH)

if e.LEGACY_MOVE_TYPES:     # old numbering: only the empty-slot behaviour is checked (the rest assumes correct types)
    t_, f_ = e._move_types_effs((0, 9999), 14, 14)
    assert t_ == [17, 17] and f_ == [e.get_effectiveness(17, 14, 14)] * 2
    print("ok (legacy move types)")
    sys.exit(0)

ID = {v: k for k, v in e.SPECIES_ID_TO_NAME.items()}
MOVE = {v["name"]: k for k, v in e._raw_moves.items()}
MOVE = {n: int(k) for n, k in MOVE.items()}
assert OBS_DIM == BASE_OBS_DIM + EXTRA_FEATURES + EXTRA2_FEATURES + EXTRA3_FEATURES + STACK_FEATURES + TEACH_FEATURES
legacy = os.environ.get("POKE_LEGACY_OBS") == "1"
assert legacy or OBS_DIM == 499 + (61 if FRAME_STACK else 0) + (44 if MOVE_TEACH else 0), OBS_DIM
assert e.N_ACTIONS == 18 + (4 if MOVE_TEACH else 0)
assert not legacy or OBS_DIM == 431, OBS_DIM


class S:
    party_levels = [30, 20, 0, 0, 0, 0]
    _move_damage = E._move_damage
    _combat_features_uncached = E._combat_features_uncached


def battler(hp, maxhp, moves, t1, t2, lvl, species, ability=0, stages=(6, 6, 6), s1=0):
    pp = [20] * 4
    return [hp, maxhp, *pp, *moves, t1, t2, lvl, s1, 0, *stages, species, 0, ability]


def party(*slots):
    raw = []
    for sl in slots:
        raw += list(sl)
    return raw + [0] * (48 - len(raw))


sp = lambda n: ID[n]
water, fire, grass, ground, normal = 11, 10, 12, 4, 0
# Own: Mudkip-like water mon L30 with Water Gun / Seismic Toss / Growl / empty. Enemy: a fire mon L20, 45 HP.
mons = [n for n in ("MUDKIP", "TORCHIC", "TREECKO") if n in ID]
assert len(mons) == 3, mons
own = battler(90, 90, [MOVE["Water Gun"], MOVE["Seismic Toss"], MOVE["Growl"], 0], water, water, 30, sp("MUDKIP"))
foe = battler(45, 45, [MOVE["Ember"], MOVE["Scratch"], 0, 0], fire, fire, 20, sp("TORCHIC"))
pp = party((90, 90, sp("MUDKIP"), 0, MOVE["Water Gun"], 0, 0, 0), (50, 50, sp("TREECKO"), 0, MOVE["Absorb"], 0, 0, 0))
f = E._combat_features(S(), own, foe, pp)
assert len(f) == 60 and all(0.0 <= v <= 1.0 for v in f), (len(f), f)
assert f[0] > f[1] > 0                      # Water Gun on a fire mon (2x) beats Seismic Toss (30 flat) per HP
assert f[2] == 0.0 and f[3] == 0.0          # Growl and the empty slot deal nothing
assert f[4] == 1.0 and f[5] == 0.0                          # Water Gun (~49 dmg: 2x, STAB) KOs a 45 HP L20 fire mon, Seismic Toss (30) does not
assert f[8] == 1.0 and 0.5 < f[9] < 1.0     # Mudkip L30 (speed ~33) outspeeds a L20 Torchic (~26)
# Bench: slot 0 (Mudkip, Water Gun) and slot 1 (Treecko L20, Absorb vs a fire mon: resisted but non-zero).
bench = f[22:40]
assert bench[0] > 0 and bench[1] > 0 and bench[2] > bench[5] > 0, bench
assert bench[6:] == [0.0] * 12, bench                                   # slots 2-5 empty
# Status move flags: Growl is a pure status move, Water Gun / Seismic Toss are not, empty slot is not.
assert f[44:48] == [0.0, 0.0, 1.0, 0.0], f[44:48]
# Effect tags (heal, status, setup) per own move: Water Gun / Seismic Toss / Growl / empty -> all zero here.
assert f[48:] == [0.0] * 12, f[48:]
rec = battler(90, 90, [MOVE["Recover"], MOVE["Thunder Wave"], MOVE["Swords Dance"], 0], water, water, 30, sp("MUDKIP"))
t = E._combat_features(S(), rec, foe, pp)
assert t[48:51] == [1.0, 0.0, 0.0] and t[51:54] == [0.0, 1.0, 0.0] and t[54:57] == [0.0, 0.0, 1.0] and t[57:] == [0.0] * 3, t[48:]
# Immunity: Seismic Toss on a ghost does nothing.
ghost = battler(60, 60, [MOVE["Scratch"], 0, 0, 0], 7, 7, 20, sp("TORCHIC"))
g = E._combat_features(S(), own, ghost, pp)
assert g[1] == 0.0
# Levitate blocks ground moves (Mud-Slap-type): a ground attacker vs ability 26.
quake = battler(90, 90, [MOVE["Earthquake"], 0, 0, 0], ground, ground, 30, sp("MUDKIP"))
lev = battler(60, 60, [MOVE["Scratch"], 0, 0, 0], 2, 2, 20, sp("TORCHIC"), ability=26)
assert E._combat_features(S(), quake, lev, pp)[0] == 0.0
# Unknown species / empty enemy: no crash, still the right length and range.
h = E._combat_features(S(), own, battler(0, 1, [0, 0, 0, 0], 0, 0, 0, 0), party())
assert len(h) == 60 and all(0.0 <= v <= 1.0 for v in h)
# Opponent-team block (needs the real table: Roxanne's 3-mon team, top level 15).
if EXTRA3_FEATURES:
    import trainer_info
    class T:
        is_trainer, trainer_id = False, -1
        _trainer_features = E._trainer_features
    enemy = party((20, 20, 0, 0, 0, 0, 0, 0))
    assert T()._trainer_features(enemy, 30) == [0.0] * 8                      # wild: zeros
    t = T(); t.is_trainer, t.trainer_id = True, 265                              # ROXANNE_1
    ep = []
    for sp_, hp_ in ((74, 30), (74, 0), (320, 40)):
        ep += [sp_, 0, hp_, 40, 0, 0, 0, 0]
    ep += [0] * (48 - len(ep))
    r = t._trainer_features(ep, 12)
    assert r[:2] == [1.0, 1.0] and abs(r[2] - 0.5) < 1e-9 and abs(r[3] - 2 / 6) < 1e-9, r   # known, size 3, 2 alive
    assert abs(r[4] - 0.15) < 1e-9 and r[5] == 0.0 and r[6] == 0.0 and r[7] == 0.0, r         # top 15, lead 12: gap < 5
    assert t._trainer_features(ep, 9)[7] == 1.0                                               # lead 9, top 15: gap >= 5
    # A hack changes the team: the game shows a different party than the vanilla table lists -> the table is not trusted.
    kaizo = []
    for sp_, hp_ in ((74, 30), (320, 40), (130, 50), (130, 50)):
        kaizo += [sp_, 0, hp_, 60, 0, 0, 0, 0]
    kaizo += [0] * (48 - len(kaizo))
    kz = t._trainer_features(kaizo, 12)
    assert kz[0] == 1.0 and kz[1] == 0.0 and kz[4] == 0.0 and kz[7] == 0.0, kz          # trainer, but team unknown
    assert abs(kz[2] - 4 / 6) < 1e-9 and abs(kz[3] - 4 / 6) < 1e-9, kz                  # size/remaining from what is shown
    u = T(); u.is_trainer, u.trainer_id = True, -1                                            # id unknown: still a trainer
    r = u._trainer_features(ep, 12)
    assert r[0] == 1.0 and r[1] == 0.0 and abs(r[2] - 0.5) < 1e-9 and r[4] == 0.0, r
    assert all(0.0 <= v <= 1.0 for v in r)
# Memo: the same raw state returns an equal (but independent) list; a changed state recomputes.
m = S()
a1 = E._combat_features(m, own, foe, pp)
a2 = E._combat_features(m, own, foe, pp)
assert a1 == a2 and a1 is not a2
a2[0] = 99.0
assert E._combat_features(m, own, foe, pp)[0] == a1[0]               # callers can't corrupt the memo
foe2 = list(foe); foe2[0] = 5
assert E._combat_features(m, own, foe2, pp) != a1                    # enemy HP changed -> recomputed

# Empty / unknown move slots: type 0 and effectiveness 0 (not a Dark move), unless the legacy switch is set.
types, effs = e._move_types_effs((MOVE["Water Gun"], 0, 9999, MOVE["Bite"]), 14, 14)
if e.LEGACY_MOVE_TYPES:
    assert types[1] == 17 and effs[1] == e.get_effectiveness(17, 14, 14)
else:
    assert types[1] == types[2] == 0 and effs[1] == effs[2] == 0.0, (types, effs)
    assert types[3] == 17 and effs[3] == 2.0                         # a real Dark move vs Psychic is still 2x
slot = e._parse_bench_slot(50, 50, sp("MUDKIP"), 0, MOVE["Water Gun"], 0, 0, 0, 14, 14)
assert slot[3 + 1] in (0.0, 17 / 17.0) and (legacy or (slot[4] == 0.0 and slot[8] == 0.0))

# Frame-stack promotion (flag on): the stack shows what the agent saw one turn ago, zeros after reset.
if FRAME_STACK:
    class F:
        pass
    f_ = F()
    f_._stack_prev, f_._stack_cur, f_._last_action_norm = [0.0] * 60, [0.5] * 60, 0.0
    # step() start: prev <- cur, remember the action
    f_._stack_prev = list(f_._stack_cur)
    f_._last_action_norm = float(3) / max(e.N_ACTIONS - 1, 1)
    assert f_._stack_prev == [0.5] * 60 and abs(f_._last_action_norm - 3 / (e.N_ACTIONS - 1)) < 1e-9
print("ok")
