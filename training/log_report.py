"""Summaries of the training logs, so nobody has to hand-write pandas for them.

    python training/log_report.py                 # everything in ../logs
    python training/log_report.py --run train_v9  # only runs whose id contains this text
    python training/log_report.py --logs <dir>

Works with old files too: a log without a `run` column is reported as run "(unknown)".
"""
import argparse
import os

import pandas as pd

GAP_BUCKETS = [("<=0", -999, 0), ("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), (">20", 21, 999)]
FLEE = ("TELEPORT", "RAN")                      # the player's own escapes (the wild mon leaving is not one)


def _read(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def _bucket_of(gap):
    for name, lo, hi in GAP_BUCKETS:
        if lo <= gap <= hi:
            return name
    return "?"


def outcome_table(df):
    """One row per group: battle count and the share of each way a battle ended."""
    def rates(g):
        n = len(g)
        decisive = g.outcome.isin(["WIN", "LOSS", "FORFEIT"]).sum()
        return pd.Series({
            "n": n,
            "win%": 100 * (g.outcome == "WIN").mean(),
            "loss%": 100 * g.outcome.isin(["LOSS", "FORFEIT"]).mean(),
            "flee%": 100 * g.outcome.isin(FLEE).mean(),
            "caught%": 100 * (g.outcome == "CAUGHT").mean(),
            "win%_of_decisive": 100 * (g.outcome == "WIN").sum() / decisive if decisive else float("nan"),
            "steps": g.steps.mean(),
            "return": g["return"].mean(),
        })
    return rates


def by_run(battles):
    if "run" not in battles.columns:
        battles = battles.assign(run="(unknown)")
    battles = battles.assign(run=battles["run"].fillna("(unknown)"))
    return battles.groupby("run").apply(outcome_table(battles), include_groups=False).round(2)


def by_gap(battles):
    wild = battles[battles.trainer_battle == 0]
    if wild.empty:
        return pd.DataFrame()
    wild = wild.assign(bucket=wild.level_gap.map(_bucket_of))
    order = [b for b, _, _ in GAP_BUCKETS]
    t = wild.groupby("bucket").apply(outcome_table(wild), include_groups=False).round(2)
    return t.reindex([b for b in order if b in t.index])


def worst_species(battles, top=8, min_n=10):
    """Enemy species with the most losses and flees (needs the enemy_species column of the newer logs)."""
    if "enemy_species" not in battles.columns:
        return pd.DataFrame()
    b = battles.dropna(subset=["enemy_species"])
    b = b[b.enemy_species != ""]
    if b.empty:
        return pd.DataFrame()
    g = b.groupby("enemy_species").apply(
        lambda x: pd.Series({"n": len(x), "loss%": 100 * x.outcome.isin(["LOSS", "FORFEIT"]).mean(),
                             "flee%": 100 * x.outcome.isin(FLEE).mean(), "gap": x.level_gap.mean()}),
        include_groups=False)
    g = g[g.n >= min_n].assign(bad=lambda d: d["loss%"] + d["flee%"]).sort_values("bad", ascending=False)
    return g.drop(columns="bad").head(top).round(2)


def time_table(profile):
    """Seconds and share of wall time per category from logs/profile.csv rows (breakdown_json = {name: [calls, seconds]})."""
    import json
    wall, parts, calls = 0.0, {}, {}
    for _, row in profile.iterrows():
        try:
            data = json.loads(row["breakdown_json"])
        except (TypeError, ValueError):
            continue
        wall += float(row["wall_s"])
        for k, (c, t) in data.items():
            parts[k] = parts.get(k, 0.0) + t
            calls[k] = calls.get(k, 0) + c
    if not wall:
        return pd.DataFrame()
    t = pd.DataFrame({"seconds": pd.Series(parts), "calls": pd.Series(calls)})
    t["share%"] = 100 * t["seconds"] / wall
    t["ms/call"] = 1000 * t["seconds"] / t["calls"].clip(lower=1)
    return t.sort_values("seconds", ascending=False).head(10).round(1)


def format_report(logs_dir, run_filter=None):
    out = []
    battles = _read(os.path.join(logs_dir, "battles.csv"))
    if battles is None or battles.empty:
        return "No battles.csv found or it is empty."
    if run_filter:
        runs = battles["run"].astype(str) if "run" in battles.columns else pd.Series("(unknown)", index=battles.index)
        battles = battles[runs.str.contains(run_filter)]
        if battles.empty:
            return f"No battles from a run matching {run_filter!r}."
    out += ["== Battles per run ==", by_run(battles).to_string(), ""]
    gap = by_gap(battles)
    if not gap.empty:
        out += ["== Wild battles by level gap ==", gap.to_string(), ""]
    ws = worst_species(battles)
    if not ws.empty:
        out += ["== Enemy species that cost the most (losses + flees, n >= 10) ==", ws.to_string(), ""]
    swaps = _read(os.path.join(logs_dir, "swaps.csv"))
    if swaps is not None and not swaps.empty:
        out += ["== Swaps by reply ==", swaps["reply"].value_counts().to_string(), ""]
    catches = _read(os.path.join(logs_dir, "catches.csv"))
    if catches is not None and not catches.empty:
        out += [f"== Catches: {len(catches)} ==", catches["reason"].fillna("manual").value_counts().to_string(), ""]
    moves = _read(os.path.join(logs_dir, "moves.csv"))
    if moves is not None and not moves.empty:
        applied = moves[moves["reply"].astype(str).str.startswith("SETMOVES:OK")]
        out += [f"== Move teaching: {len(moves)} attempts, {len(applied)} applied, "
                f"mean score gain {applied['delta_score'].mean():.1f} ==" if len(applied) else
                f"== Move teaching: {len(moves)} attempts, none applied ==", ""]
    profile = _read(os.path.join(logs_dir, "profile.csv"))
    if profile is not None and not profile.empty:
        if run_filter and "run" in profile.columns:
            profile = profile[profile["run"].astype(str).str.contains(run_filter)]
        tt = time_table(profile) if not profile.empty else pd.DataFrame()
        if not tt.empty:
            out += ["== Where the time goes (profile.csv; shares are of battle wall time) ==", tt.to_string(), ""]
    runs = _read(os.path.join(logs_dir, "runs.csv"))
    if runs is not None and not runs.empty:
        out += ["== Runs ==", runs[["timestamp", "run", "obs_dim", "n_actions", "git", "flags"]].tail(8).to_string(), ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"))
    ap.add_argument("--run", default=None, help="only runs whose id contains this text")
    args = ap.parse_args()
    print(format_report(args.logs, args.run))


if __name__ == "__main__":
    main()
