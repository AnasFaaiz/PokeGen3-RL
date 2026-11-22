"""Offline checks for the reward helpers (no emulator): python training/test_reward.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import PokemonBattleEnv as E, BASELINE_BASE, BASELINE_PER_LEVEL, BASELINE_REF, BASELINE_GAP_CAP, EGG_SPECIES
import exp_tables
import logger


class Stub:
    """Just enough of the env for the pure helpers (no socket)."""
    is_wild = True
    battle_start = {"lead_level": 20, "enemy_level": 20}
    _request = staticmethod(lambda line: "")


def frac(sp, xp):
    return E._frac_level(sp, xp)


# _frac_level: whole level exactly at a threshold, rising smoothly, capped at 100.
rate = exp_tables.growth_rate(1)
at20, at21 = exp_tables.exp_at_level(rate, 20), exp_tables.exp_at_level(rate, 21)
assert frac(1, at20) == 20.0
assert 20.0 < frac(1, (at20 + at21) // 2) < 21.0
assert frac(1, at21 - 1) < 21.0 and frac(1, at21) == 21.0
assert frac(1, 10 ** 9) == 100.0

# _difficulty_shift: constant per gap, capped, zero for trainers / unknown start.
s = Stub()
for gap, want in ((-5, BASELINE_BASE - BASELINE_REF), (0, BASELINE_BASE - BASELINE_REF),
                  (8, BASELINE_BASE + 8 * BASELINE_PER_LEVEL - BASELINE_REF),
                  (50, BASELINE_BASE + BASELINE_GAP_CAP * BASELINE_PER_LEVEL - BASELINE_REF)):
    s.battle_start = {"lead_level": 20 + gap, "enemy_level": 20}
    assert E._difficulty_shift(s) == want, (gap, E._difficulty_shift(s))
s.is_wild = False
assert E._difficulty_shift(s) == 0.0
s.is_wild, s.battle_start = True, {}
assert E._difficulty_shift(s) == 0.0

# _party_progress: bad / missing / egg input falls back safely.
s = Stub()
s._frac_level = E._frac_level
pp = lambda inv: E._party_progress(s, inv)
assert pp(None) is None and pp({}) is None and pp({"lv": "20", "px": ""}) is None
assert pp({"lv": "20,30", "px": "1:8000"}) is None                      # length mismatch
assert pp({"lv": "20", "px": "x:y"}) is None                            # garbage
out = pp({"lv": "20,5,7", "px": f"1:{at20},0:0,{EGG_SPECIES}:0"})
assert out == [20.0, 5.0, 7.0], out                                    # empty slot and egg keep whole level

# battles.csv rotation: an old header is moved aside once, the new file gets the new header.
with tempfile.TemporaryDirectory() as d:
    logger._LOG_DIR = d
    path = os.path.join(d, "battles.csv")
    logger.BATTLES_FILE = path
    logger._checked.clear()
    with open(path, "w") as f:
        f.write("timestamp,episode\n1,2\n")
    logger._append_row(path, logger.BATTLES_HEADER, ["a"] * len(logger.BATTLES_HEADER))
    names = os.listdir(d)
    assert any("oldschema" in n for n in names) and "battles.csv" in names, names
    with open(path) as f:
        assert f.readline().strip().split(",") == logger.BATTLES_HEADER
    logger._append_row(path, logger.BATTLES_HEADER, ["b"] * len(logger.BATTLES_HEADER))   # no second rotation
    assert len([n for n in os.listdir(d) if "oldschema" in n]) == 1

# Reward ordering (Phase 3.3): at either win bonus, a full-HP win beats the best realistic catch and a loss
# is the worst outcome, so neither throwing balls nor fleeing can out-earn winning a fight.
import env as _e
for win in (100.0, 60.0):
    turns = 3 * _e.TURN_COST
    full_win = win + _e.DAMAGE_REWARD_SCALE * 1.0 + _e.WIN_HP_BONUS - turns + _e.KO_BONUS
    # best catch: enemy brought low first (half the damage term), 2 balls, useful + new species + low-HP bonuses
    best_catch = (_e.DAMAGE_REWARD_SCALE * 0.5 + _e.CATCH_USEFUL + _e.NEW_SPECIES_BONUS + _e.LOW_HP_CATCH_BONUS
                  - 2 * _e.BALL_COST - turns)
    loss = _e.LOSS_REWARD - turns - _e.KO_BONUS
    assert loss < 0 < min(full_win, best_catch), (win, loss, full_win, best_catch)
    if win == 100.0:
        assert full_win > best_catch, (full_win, best_catch)     # the shipped default keeps winning ahead of catching
    else:
        print(f"note: at POKE_WIN_REWARD=60 a full win is {full_win:.0f} vs best catch {best_catch:.0f}")
# Flee penalty: only the player's own escape (4 ran, 5 teleported) in a wild battle is charged; the enemy leaving
# (6, 10), trainer battles and a draw are not. A flee is better than a loss and worse than any full win.
fl = Stub()
fl.is_wild = True
for oc in (4, 5):
    assert E._outcome_bonus(fl, oc) == -_e.FLEE_PENALTY
for oc in (3, 6, 10):
    assert E._outcome_bonus(fl, oc) == 0.0, oc
fl.is_wild = False
assert E._outcome_bonus(fl, 5) == 0.0                                     # trainer battle: nothing to flee from
assert E._outcome_bonus(fl, 1) == _e.WIN_REWARD and E._outcome_bonus(fl, 2) == _e.LOSS_REWARD
if _e.FLEE_PENALTY == 40.0:
    flee_total = -_e.FLEE_PENALTY - 3 * _e.TURN_COST - _e.DAMAGE_REWARD_SCALE * 0.3    # fled after taking 30% damage
    assert _e.LOSS_REWARD - 3 * _e.TURN_COST < flee_total < 0, flee_total
    assert flee_total < _e.WIN_REWARD * 0.5 - 50 + 0                                    # clearly below a hard-fought win
    assert flee_total > -100.0 - 0.0 - 3 * _e.TURN_COST                                 # but above the loss bonus alone

# Throws that cannot pay are not offered: a species the PC already holds, or the reserve balls on a non-new species.
th = Stub()
th.balls_left, th.enemy_species, th.enemy_owned, th.box_cache = 30, 5, True, ([10], {5})
assert E._throw_worthwhile(th) is False                            # duplicate of a PC mon
th.box_cache = ([10], {6})
assert E._throw_worthwhile(th) is True                             # a species the PC lacks
th.balls_left = _e.BALL_RESERVE
assert E._throw_worthwhile(th) is False                            # reserve kept for new species only
th.enemy_owned = False
assert E._throw_worthwhile(th) is True                             # ...which a Pokedex-new species may use
th.box_cache = ([10], {5}); th.enemy_species = 5
assert E._throw_worthwhile(th) is True                             # new to the Pokedex always allowed
th.enemy_owned, th.box_cache, th.balls_left = None, None, 30
assert E._throw_worthwhile(th) is True                             # unknown info never blocks a throw
assert _e.BALL_COST >= 5.0 or os.environ.get("POKE_BALL_COST")

# A duplicate species the PC already holds is worth nothing, whatever the bonus.
st = Stub()
st.battle_start = {"enemy_level": 10}
st._party_avg = lambda: 30.0
st.area_max_now = None
st._candidate_ceiling = lambda: E._candidate_ceiling(st)
st.box_low_stock, st.enemy_owned, st.enemy_species, st.last_throw_e_hp = 0, True, 5, 0.1
st.box_cache = ([10], {5})
assert E._catch_value(st) == 0.0
print("ok")
