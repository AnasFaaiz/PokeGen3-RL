"""Offline checks for catch/swap interplay and the TensorBoard bucket filter: python training/test_policies.py"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catch_policy import CatchContext, CatchPolicy, CANDIDATE_MARGIN
from swap_policy import SwapPolicy, BoxEntry
import env as e
from env import PokemonBattleEnv as E

# --- catching is stock-driven: a party sitting near the area level still refills the box.
p = CatchPolicy()
for lv in [20] * 10:
    p.observe_area("0.32", lv)
area = p.area_max("0.32")
assert area == 20
base = dict(enemy_species=288, enemy_level=21, party_levels=[24, 23, 22, 22, 21, 22], box_levels=[60, 55],
            box_species={25}, area_max=area, trainer=False, balls=10)
assert p.decide(CatchContext(**base)), p.last_reason                                   # near-area party catches
assert not p.decide(CatchContext(**{**base, "enemy_level": area + CANDIDATE_MARGIN + 1}))
assert p.last_reason == "enemy too high"
assert not p.decide(CatchContext(**{**base, "box_levels": [18, 19, 20, 21, 22, 23]}))   # stock already full
assert not CatchPolicy(stock_driven=False).decide(CatchContext(**base))                 # the old rule would refuse

# --- _catch_value: "useful" is the swap candidate band when the area is known, else the old party-average rule.
st = SimpleNamespace(battle_start={"enemy_level": 22}, area_max_now=20, box_low_stock=0, enemy_owned=None,
                     enemy_species=1, box_cache=None, last_throw_e_hp=1.0)
st._party_avg = lambda: 26.0
st._candidate_ceiling = lambda: E._candidate_ceiling(st)
assert E._candidate_ceiling(st) == 20 + CANDIDATE_MARGIN
assert E._catch_value(st) == e.CATCH_USEFUL                                              # L22 <= 25
st.battle_start["enemy_level"] = 27
assert E._catch_value(st) == e.CATCH_PLAIN                                               # above the band
st.area_max_now = None                                                                   # fallback: avg - gap
assert E._candidate_ceiling(st) == 26.0 - e.CANDIDATE_GAP
st.battle_start["enemy_level"] = 10
assert E._catch_value(st) == e.CATCH_USEFUL
st.box_low_stock = e.STOCK_WANTED                                                        # stock full: reduced payout
assert E._catch_value(st) == e.CATCH_USEFUL * e.STOCK_FULL_FACTOR

# --- swap hysteresis: with the default gap, nothing a pass brings in is swappable again straight away.
sp = SwapPolicy(seed=0, keep_lead_prob=0.0)
box = [BoxEntry(f"0.{i}", 280 + i, lv) for i, lv in enumerate((18, 19, 20, 21, 22, 23, 24, 25, 26))]
for ep in range(50):
    sp.last_swap_episode = -10 ** 9
    for slot, ent in sp.plan(ep, [40, 40, 40], 20, box):
        assert ent.level < 20 + sp.gap_to_swap, (ent, "would be overleveled immediately")
        assert not sp.overleveled_slots([ent.level], 20), ent

# --- TensorBoard callback: wild battles fill the gap buckets, trainer fights are tracked apart.
import train_dqn as t
logged = {}
cb = t.WinRateCallback()
cb.model = SimpleNamespace(logger=SimpleNamespace(record=lambda k, v: logged.__setitem__(k, v)))
def feed(outcome, gap, trainer, turns=3, ret=100.0):
    cb.locals = {"dones": np.array([True]),
                 "infos": [{"outcome": outcome, "battle_stats": {"level_gap": gap, "trainer_battle": trainer,
                                                                  "turns": turns, "return": ret}}]}
    cb._on_step()
for _ in range(12):
    feed(1, 3, 0)                       # wild wins at gap 3
for _ in range(6):
    feed(2, 3, 1)                       # trainer losses at gap 3
cb._on_rollout_end()
assert logged["winrate_by_gap/le5"] == 100.0, logged                                 # trainer losses not mixed in
assert logged["winrate_trainer"] == 0.0 and logged["turns_by_gap/le5"] == 3.0
assert "winrate_by_gap/gt10" not in logged                                              # needs >= 10 results
cb2 = t.WinRateCallback(); logged2 = {}
cb2.model = SimpleNamespace(logger=SimpleNamespace(record=lambda k, v: logged2.__setitem__(k, v)))
cb = cb2
for _ in range(4):
    feed(2, 3, 1)
cb._on_rollout_end()
assert "winrate_trainer" not in logged2                                                  # fewer than 5 trainer results
# --- flee/all-win metrics: a flee counts as not-won in winrate_all_by_gap and shows up in flee_rate_by_gap.
logged3 = {}
cb3 = t.WinRateCallback()
cb3.model = SimpleNamespace(logger=SimpleNamespace(record=lambda k, v: logged3.__setitem__(k, v)))
cb = cb3
for _ in range(12):
    feed(1, 2, 0)                       # 12 wild wins at gap 2
for _ in range(3):
    feed(5, 2, 0)                       # 3 player teleports at gap 2
feed(10, 2, 0)                          # enemy teleported: neither a win nor a flee
feed(7, 2, 0)                           # caught: excluded
cb._on_rollout_end()
assert logged3["winrate_by_gap/le5"] == 100.0                                # decisive fights only: 12/12
assert logged3["winrate_all_by_gap/le5"] == 80.0 and logged3["flee_rate_by_gap/le5"] == 20.0, logged3

# --- teleported flag is the player's own escape only.
for oc, want in ((4, 1), (5, 1), (6, 0), (10, 0), (1, 0), (2, 0)):
    assert int(oc in e.PLAYER_ESCAPE_OUTCOMES) == want, oc

# --- entropy schedule: starts at `start`, ends at `end`, never increases, clamps beyond the total.
an = t.EntCoefAnneal(start=0.01, end=0.002, total=1000)
vals = [an.value(x) for x in (0, 250, 500, 1000, 5000)]
assert vals[0] == 0.01 and abs(vals[3] - 0.002) < 1e-12 and vals[4] == vals[3]
assert all(a >= b for a, b in zip(vals, vals[1:]))
an.model = SimpleNamespace(num_timesteps=24_576, ent_coef=0.05)
an.logger_ = None
an.__class__.logger = property(lambda self: SimpleNamespace(record=lambda k, v: None))
an._on_training_start()
assert an.model.ent_coef == 0.01 and an.t0 == 24_576                           # progress counts from this run's start
an.model.num_timesteps = 24_576 + 500
an._on_rollout_end()
assert abs(an.model.ent_coef - 0.006) < 1e-9, an.model.ent_coef

# --- battle_stats carries text for the logs (species, map) and "" for an unknown area: the numeric means still work.
logged5 = {}
cb5 = t.WinRateCallback()
cb5.model = SimpleNamespace(logger=SimpleNamespace(record=lambda k, v: logged5.__setitem__(k, v)))
cb = cb5
for i in range(5):
    cb.locals = {"dones": np.array([True]),
                 "infos": [{"outcome": 1, "battle_stats": {"level_gap": 4 + i, "trainer_battle": 0, "turns": 6, "return": 100.0,
                                                          "lead_species": "MUDKIP", "enemy_species": "ZIGZAGOON", "map": "0.32",
                                                          "area_max": "" if i % 2 else 20, "faints": 0, "shift": float("nan")}}]}
    cb._on_step()
cb._on_rollout_end()                      # used to raise TypeError at the first update of a real run
assert logged5["battle/level_gap_mean"] == 6.0 and logged5["battle/turns_mean"] == 6.0 and logged5["battle/area_max_mean"] == 20.0
assert not any(k.endswith("species_mean") or k == "battle/map_mean" or k == "battle/shift_mean" for k in logged5), logged5

# --- checkpoint suffix follows the effective layout (legacy mode switches teaching and stacking off).
assert t.model_suffix(False, False) == "" and t.model_suffix(True, False) == "_stack"
assert t.model_suffix(False, True) == "_teach" and t.model_suffix(True, True) == "_stack_teach"
assert t.model_suffix(t.FRAME_STACK, t.MOVE_TEACH) == (("_stack" if e.FRAME_STACK else "") + ("_teach" if e.MOVE_TEACH else ""))
if os.environ.get("POKE_LEGACY_OBS") == "1":
    assert t.model_suffix(t.FRAME_STACK, t.MOVE_TEACH) == ""

# --- run_eval bucket maths: flees are dropped from win_rate but counted in win_rate_incl_flee.
import importlib.util
spec = importlib.util.spec_from_file_location("run_eval", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                        "..", "analysis", "run_eval.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
rows = [(3, "win", 4, 80.0, 0), (4, "win", 6, 60.0, 1), (2, "loss", 20, 0.0, 6), (1, "flee", 5, 50.0, 0),
        (12, "win", 2, 100.0, 0)]
res = {r["bucket"]: r for r in ev.summarise(rows)}
assert res["1-5"]["n"] == 4 and abs(res["1-5"]["win_rate"] - 200 / 3) < 1e-9          # 2 wins of 3 decisive
assert res["1-5"]["flee_pct"] == 25.0 and res["1-5"]["win_rate_incl_flee"] == 50.0
assert res[">10"]["win_rate"] == 100.0 and res["all"]["n"] == 5 and "<=0" not in res
assert abs(res["1-5"]["mean_turns"] - (4 + 6 + 20) / 3) < 1e-9                         # flees excluded from fight stats
print("ok")
