"""Offline checks for the logging changes: python training/test_logs.py"""
import csv
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logger as L
import log_report

with tempfile.TemporaryDirectory() as d:
    for name in ("LOG_FILE", "CSV_FILE", "BATTLES_FILE", "SWAP_FILE", "CATCH_FILE", "LEVEL_GAP_FILE", "RUNS_FILE",
                 "MOVES_FILE", "PROFILE_FILE"):
        setattr(L, name, os.path.join(d, os.path.basename(getattr(L, name))))
    L._LOG_DIR = d
    L._checked.clear()

    # outcome names follow gBattleOutcome
    assert [L.outcome_name(o) for o in (1, 2, 3, 4, 5, 6, 7, 10)] == [
        "WIN", "LOSS", "DRAW", "RAN", "TELEPORT", "FLED", "CAUGHT", "ENEMY_TELEPORT"]
    assert L.outcome_name(99) == "END(99)"

    # run id: readable suffix, in every row and in runs.csv
    rid = L.set_run("unit")
    assert rid.endswith("_unit") and L.RUN_ID == rid
    L.log_run_start("test", 499, 18, "v9")
    L.log_swap(1, "0.32", 20, 0, 31, "ODDISH", 20, "SWAP:OK")
    L.log_catch(1, "ODDISH", 20, 25, 20, 1, "low-level stock")
    L.log_move(1, 0, "MUDKIP", 30, "Growl", "Water Gun", 12.3, "levelup", "SETMOVES:OK")
    for name in ("runs.csv", "swaps.csv", "catches.csv", "moves.csv"):
        rows = list(csv.reader(open(os.path.join(d, name))))
        assert "run" in rows[0] and rows[1][rows[0].index("run")] == rid, name

    # an older header is rotated aside for every log, not only battles.csv
    with open(L.SWAP_FILE, "w") as f:
        f.write("timestamp,episode\n1,2\n")
    L._checked.clear()
    L.log_swap(2, "0.32", 20, 0, 31, "LEDIAN", 20, "SWAP:OK")
    assert any("oldschema" in n and n.startswith("swaps") for n in os.listdir(d))
    assert open(L.SWAP_FILE).readline().strip().split(",") == [
        "timestamp", "run", "episode", "map", "area_max", "party_slot", "out_level", "in_species", "in_level", "reply"]

    # observation: 4 decimals, no spaces, and only every Nth step when asked
    L.init_csv()
    obs = np.array([0.123456789, 1.0, 0.5], dtype=np.float32)
    L.LOG_OBS_EVERY, L._obs_count = 1, 0
    L.log_step(1, 0, 3, 1.0, False, False, obs=obs)
    row = list(csv.reader(open(L.CSV_FILE)))[1]
    assert row[-1] == "[0.1235,1.0,0.5]" and json.loads(row[-1])[0] == 0.1235, row[-1]
    L.LOG_OBS_EVERY, L._obs_count = 3, 0
    for s in range(6):
        L.log_step(1, s + 1, 3, 1.0, False, False, obs=obs)
    kept = [r[-1] != "" for r in list(csv.reader(open(L.CSV_FILE)))[2:]]
    assert kept == [True, False, False, True, False, False], kept
    L.LOG_OBS_EVERY = 1

    # size rotation: the big file moves to .1, a new one with a header appears, at most KEEP_OLD_LOGS old files
    L.BIG_CSV_BYTES, L.BIG_TXT_BYTES = 200, 200
    L._write_count = 199
    L.log_step(2, 0, 1, 1.0, False, False, obs=obs)             # 200th write triggers the size check
    assert os.path.exists(L.CSV_FILE + ".1") and os.path.exists(L.LOG_FILE + ".1")
    assert open(L.CSV_FILE).readline().strip().split(",") == L.CSV_HEADER
    for i in range(5):
        with open(L.CSV_FILE, "a") as f:
            f.write("x" * 300 + "\n")
        L._write_count = 199
        L.log_step(3 + i, 0, 1, 1.0, False, False, obs=obs)
    olds = sorted(n for n in os.listdir(d) if n.startswith("battle_data.csv."))
    assert olds == ["battle_data.csv.1", "battle_data.csv.2"], olds
    assert not L._rotate_if_big(os.path.join(d, "missing.csv"), 10)    # a missing file is not an error

    # battles.csv row has the new columns and the report reads it (old rows without a run are '(unknown)')
    L.BATTLES_FILE = os.path.join(d, "battles.csv")
    stats = {"trainer_battle": 0, "lead_level": 30, "enemy_level": 20, "level_gap": 10, "moves": 3,
             "voluntary_switches": 0, "forced_switches": 0, "failed_switches": 0, "balls_thrown": 0, "caught": 0,
             "hp_left_pct": 80.0, "faints": 0, "start_hp_pct": 100.0, "trainer_id": -1,
             "lead_species": "MUDKIP", "enemy_species": "ZIGZAGOON", "map": "0.32", "area_max": 20}
    for ep, oc in enumerate((1, 1, 1, 2, 5, 7), start=1):
        L.log_battle_row(ep, oc, 4, 100.0 if oc == 1 else -50.0, stats)
    head = open(L.BATTLES_FILE).readline().strip().split(",")
    assert head[-5:] == ["run", "lead_species", "enemy_species", "map", "area_max"] and len(head) == len(L.BATTLES_HEADER)
    rep = log_report.format_report(d)
    assert "unit" in rep and "Wild battles by level gap" in rep and "== Swaps by reply ==" in rep, rep
    assert "Move teaching: 1 attempts, 1 applied, mean score gain 12.3" in rep, rep
    only = log_report.format_report(d, run_filter="nomatch")
    assert "No battles from a run matching" in only
    t = log_report.by_gap(log_report.pd.read_csv(L.BATTLES_FILE))
    assert t.loc["6-10", "n"] == 6 and abs(t.loc["6-10", "win%"] - 50.0) < 1e-9       # 3 wins / 6
    assert abs(t.loc["6-10", "flee%"] - 100 / 6) < 0.01 and abs(t.loc["6-10", "win%_of_decisive"] - 75.0) < 1e-9
# --- profile.csv + the time table, env tag and prepare_logs (worker setup)
with tempfile.TemporaryDirectory() as d:
    for name in ("LOG_FILE", "CSV_FILE", "BATTLES_FILE", "SWAP_FILE", "CATCH_FILE", "LEVEL_GAP_FILE", "RUNS_FILE",
                 "MOVES_FILE", "PROFILE_FILE"):
        setattr(L, name, os.path.join(d, os.path.basename(getattr(L, name))))
    L._checked.clear()
    L.set_env("20261122_000000_train_e2", "e2")
    L.log_profile(50, 25, 600.0, {"nav_to_battle": [25, 200.0], "cmd:GET_STATE": [900, 90.5], "swap_check": [25, 12.0]})
    L.log_profile(75, 25, 400.0, {"nav_to_battle": [25, 100.0], "cmd:GET_STATE": [700, 70.0]})
    prof = log_report.pd.read_csv(L.PROFILE_FILE)
    assert prof.loc[0, "run"].endswith("_e2") and prof.shape[0] == 2
    tt = log_report.time_table(prof)
    assert tt.index[0] == "nav_to_battle" and abs(tt.loc["nav_to_battle", "seconds"] - 300.0) < 1e-6
    assert abs(tt.loc["nav_to_battle", "share%"] - 30.0) < 0.1 and tt.loc["cmd:GET_STATE", "calls"] == 1600
    # worker log lines carry the env tag; the main process (no tag) does not
    L.init_csv()
    L.log_step(1, 0, 3, 1.0, False, False)
    assert "[e2] Ep 001" in open(L.LOG_FILE).read()
    L.set_env("main", "")
    L.log_step(2, 0, 3, 1.0, False, False)
    assert open(L.LOG_FILE).read().splitlines()[-1].split("]")[1].lstrip().startswith("Ep 002")
    # prepare_logs creates the header once and moves an old-header file aside, so workers only append
    with open(L.BATTLES_FILE, "w") as f:
        f.write("timestamp,episode\n1,2\n")
    os.remove(L.CSV_FILE)
    L._checked.clear()
    L.prepare_logs()
    assert open(L.CSV_FILE).readline().strip().split(",") == L.CSV_HEADER
    assert any(n.startswith("battles_oldschema") for n in os.listdir(d))
print("ok")
