import csv
import os
import json
import subprocess
from datetime import datetime

# Identifies this process in every log row (and in logs/runs.csv), so rows from different runs or settings can
# be told apart. train_dqn.py / run_eval.py call set_run() with a readable name; the default is the start time.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def set_run(name):
    global RUN_ID
    RUN_ID = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}" if name else RUN_ID
    return RUN_ID

ENV_TAG = ""      # "e2" in a worker of a multi-env run; prefixed to the human-readable log lines


def set_env(run_id, tag):
    """Used by training workers: a unique run id per env (so rows from different windows can be told apart) and a tag."""
    global RUN_ID, ENV_TAG
    RUN_ID, ENV_TAG = run_id, tag

# Size limits: the per-step CSV stores a 400+ value observation per row and grows by hundreds of MB.
BIG_CSV_BYTES = int(float(os.environ.get("POKE_LOG_CSV_MAX_MB", 200)) * 1024 * 1024)
BIG_TXT_BYTES = int(float(os.environ.get("POKE_LOG_TXT_MAX_MB", 20)) * 1024 * 1024)
LOG_OBS_EVERY = max(int(os.environ.get("POKE_LOG_OBS_EVERY", 1)), 1)   # keep the observation on every Nth step only
KEEP_OLD_LOGS = 2
_write_count = 0
_obs_count = 0


def _rotate_if_big(path, limit, keep=KEEP_OLD_LOGS):
    """path -> path.1 -> path.2 ... when it passes `limit` bytes; the oldest beyond `keep` is deleted."""
    try:
        if os.path.getsize(path) < limit:
            return False
    except OSError:
        return False
    for i in range(keep, 0, -1):
        src = path if i == 1 else f"{path}.{i - 1}"
        dst = f"{path}.{i}"
        if os.path.exists(src):
            if i == keep and os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
    return True

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(_LOG_DIR, "battle_log.txt")
CSV_FILE = os.path.join(_LOG_DIR, "battle_data.csv")

# Predefined action names mapping for 17 discrete actions
TARGET_NAMES = ["Enemy Left", "Enemy Right", "Partner"]

def decode_action(action_idx, move_names=None, party_names=None):
    """
    Decodes discrete action 0-17 into a human-readable action description.
    move_names: list of up to 4 move strings for P1 [m1, m2, m3, m4]
    party_names: list of bench Pokemon names for slots 1-5
    """
    if action_idx < 12:
        move_slot = action_idx % 4
        target_slot = action_idx // 4
        m_name = move_names[move_slot] if move_names and move_slot < len(move_names) else f"Move_{move_slot+1}"
        t_name = TARGET_NAMES[target_slot] if target_slot < len(TARGET_NAMES) else f"Target_{target_slot}"
        return f"{m_name} -> {t_name}"
    elif action_idx == 17:
        return "THROW BALL"
    else:
        bench_slot = action_idx - 11
        p_name = party_names[bench_slot - 1] if party_names and (bench_slot - 1) < len(party_names) else f"Slot_{bench_slot}"
        return f"SWITCH -> {p_name}"

REJECT_FILE = os.path.join(_LOG_DIR, "rejects.csv")

def log_reject(episode, step, action, reply, phase, mask, retry, substituted_with=None):
    """One row per refused ACTION so the missing mask rule can be found.
    reply is the bridge's REJECT:<reason>; phase is the bridge phase at the
    time; mask is the 18-bool legal-action mask the policy was given."""
    new = not os.path.exists(REJECT_FILE)
    with open(REJECT_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "episode", "step", "action", "reply", "phase",
                        "mask_legal_at_action", "mask", "retry", "substituted_with"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), episode, step, action, reply,
                    phase, int(bool(mask[action])) if mask is not None else "",
                    "".join("1" if m else "0" for m in mask) if mask is not None else "",
                    retry, "" if substituted_with is None else substituted_with])

LEVEL_GAP_FILE = os.path.join(_LOG_DIR, "level_gap.csv")

LEVEL_GAP_HEADER = ["timestamp", "run", "episode", "party_levels", "box_top5_levels", "balls", "money"]
CATCH_HEADER = ["timestamp", "run", "episode", "species", "level", "party_min_level", "area_max", "balls_used", "reason"]
SWAP_HEADER = ["timestamp", "run", "episode", "map", "area_max", "party_slot", "out_level", "in_species", "in_level", "reply"]


def log_level_gap(episode, party_levels, box_top, balls, money):
    """Periodic roster snapshot: party levels vs the strongest PC mons."""
    _append_row(LEVEL_GAP_FILE, LEVEL_GAP_HEADER,
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), RUN_ID, episode,
                 " ".join(map(str, party_levels)), " ".join(map(str, box_top)), balls, money])

CATCH_FILE = os.path.join(_LOG_DIR, "catches.csv")

def log_catch(episode, species, level, party_min_level, area_max, balls_used, reason):
    """One row per caught Pokemon. reason is the scripted rule's reason, or
    empty for a catch the rule did not start (a manual throw)."""
    _append_row(CATCH_FILE, CATCH_HEADER,
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), RUN_ID, episode, species, level,
                 party_min_level, area_max, balls_used, reason])

SWAP_FILE = os.path.join(_LOG_DIR, "swaps.csv")

def log_swap(episode, map_key, area_max, party_slot, out_level, in_species, in_level, reply):
    """One row per PC swap attempt."""
    _append_row(SWAP_FILE, SWAP_HEADER,
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), RUN_ID, episode, map_key, area_max,
                 party_slot, out_level, in_species, in_level, reply])

def alert_loss():
    print("[ALERT] Battle lost!")
    try:
        subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/message.oga"], check=False)
    except FileNotFoundError:
        os.system('echo -e "\\a"')

CSV_HEADER = [
    "timestamp", "episode", "step", "action_id", "action_desc",
    "p1_hp_pct", "e1_hp_pct", "p2_hp_pct", "e3_hp_pct",
    "eff_mult", "chosen_prob", "num_legal_actions",
    "reward", "terminated", "player_lost", "turn_timeout",
    "p1_species", "p1_level",
    "e1_species", "e1_level", "e1_status", "e1_atk_buff", "e1_used_move",
    "obs_json",
]

BATTLES_FILE = os.path.join(_LOG_DIR, "battles.csv")
BATTLES_HEADER = [
    "timestamp", "episode", "outcome", "trainer_battle", "lead_level", "enemy_level", "level_gap",
    "steps", "return", "moves", "voluntary_switches", "forced_switches", "failed_switches",
    "balls_thrown", "caught", "hp_left_pct", "faints", "start_hp_pct", "trainer_id",
    "run", "lead_species", "enemy_species", "map", "area_max",
]
THROWS_FILE = os.path.join(_LOG_DIR, "throws.csv")
THROWS_HEADER = [
    "timestamp", "episode", "map", "species", "level", "ball_number", "result",
    "area_max", "party_min_level", "reason",
]


def _append_row(path, header, row):
    if os.path.exists(path):
        _rotate_if_old_header(path, header)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _rotate_if_old_header(path, header):
    """Move a file with an older header aside so schemas never mix (checked once per process)."""
    if path in _checked:
        return
    _checked.add(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip().split(",")
    if first != header:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(os.path.basename(path))
        os.rename(path, os.path.join(os.path.dirname(path), f"{base}_oldschema_{stamp}{ext}"))
        print(f"[logger] {os.path.basename(path)} had an older header; moved aside")


_checked = set()


def init_csv():
    """Create battle_data.csv with the current header. If an existing file has a
    different (older) header, rename it aside instead of mixing schemas."""
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline().strip().split(",")
        if first == CSV_HEADER:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aside = os.path.join(_LOG_DIR, f"battle_data_oldschema_{stamp}.csv")
        os.rename(CSV_FILE, aside)
        print(f"[logger] battle_data.csv had an older header; moved to {os.path.basename(aside)}")
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def log_battle_row(episode, outcome, steps, total_return, stats):
    """One row per finished battle. stats is the env's battle_stats dict."""
    _append_row(BATTLES_FILE, BATTLES_HEADER, [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), episode, outcome_name(outcome),
        stats.get("trainer_battle"), stats.get("lead_level"), stats.get("enemy_level"),
        stats.get("level_gap"), steps, round(total_return, 2), stats.get("moves"),
        stats.get("voluntary_switches"), stats.get("forced_switches"), stats.get("failed_switches"),
        stats.get("balls_thrown"), stats.get("caught"),
        round(stats.get("hp_left_pct", 0.0), 1), stats.get("faints"),
        round(stats.get("start_hp_pct", 0.0), 1), stats.get("trainer_id", -1),
        RUN_ID, stats.get("lead_species", ""), stats.get("enemy_species", ""), stats.get("map", ""),
        stats.get("area_max", ""),
    ])


def log_throw(episode, map_key, species, level, ball_number, result, area_max, party_min, reason):
    """One row per scripted throw: OK (ball thrown, battle continues), CAUGHT,
    FAILED, REJECTED:<why>, or the battle outcome if it ended another way."""
    _append_row(THROWS_FILE, THROWS_HEADER, [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), episode, map_key, species, level,
        ball_number, result, area_max, party_min, reason,
    ])


def log_step(
    episode,
    step,
    action,
    reward,
    terminated,
    player_lost,
    obs_dict=None,
    action_meta=None,
    obs=None,
    turn_timeout=False
):
    """
    Comprehensive logger for turn-by-turn battle analysis.
    obs_dict: dict containing key state metrics (hp percentages, active species, effectiveness)
    action_meta: dict containing policy info (prob, legal_actions_count, action_str)
    obs: the FULL raw observation array/list for this step (e.g. the 285-float
         vector from env.py's PokemonBattleEnv._get_state()). Optional -- if
         omitted, obs_json is written as an empty string and this row simply
         won't be usable by distill_policy.py (everything else still works
         exactly as before). Pass it from env.py's step() to enable distillation.
    turn_timeout: True when this row represents a turn that didn't resolve
         normally (env.py's step() "not turn_resolved" branch) -- lets you
         filter/count how often turns are timing out from the CSV directly.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Defaults if not supplied
    obs_dict = obs_dict or {}
    action_meta = action_meta or {}

    p1_hp = obs_dict.get("p1_hp", 0.0)
    e1_hp = obs_dict.get("e1_hp", 0.0)
    p2_hp = obs_dict.get("p2_hp", 0.0)
    e3_hp = obs_dict.get("e3_hp", 0.0)
    eff = obs_dict.get("eff_mult", 1.0)

    p1_species = obs_dict.get("p1_species", "Unknown")
    p1_level = obs_dict.get("p1_level", 0)

    e1_species = obs_dict.get("e1_species", "Unknown")
    e1_level = obs_dict.get("e1_level", 0)
    e1_status = obs_dict.get("e1_status", 0)
    e1_atk_buff = obs_dict.get("e1_atk_buff", 0)
    e1_used_move = obs_dict.get("e1_used_move", "-")

    action_desc = action_meta.get("desc", f"Action_{action}")
    action_prob = action_meta.get("prob", 0.0)
    legal_count = action_meta.get("legal_count", 18)

    # Serialize the full observation (if provided) for post-hoc policy
    # distillation (see distill_policy.py). Kept as its own column so
    # every other column/row shape stays unchanged for existing readers.
    global _write_count, _obs_count
    obs_json = ""
    _obs_count += 1
    if obs is not None and (_obs_count - 1) % LOG_OBS_EVERY == 0:
        try:
            obs_list = obs.tolist() if hasattr(obs, "tolist") else list(obs)
            # 4 decimals and no spaces: about a third of the size of full-precision floats, still plenty for
            # distillation (the observation is scaled to [0, 1]).
            obs_json = json.dumps([round(float(x), 4) for x in obs_list], separators=(",", ":"))
        except Exception:
            obs_json = ""

    # 1. Clean, human-readable terminal/log file string
    log_line = (
        f"[{timestamp}]{' [' + ENV_TAG + ']' if ENV_TAG else ''} Ep {episode:03d} | Step {step:02d} | "
        f"{action_desc.ljust(26)} | "
        f"Prob: {action_prob*100:5.1f}% | "
        f"Eff: {eff:3.1f}x | "
        f"P1: {p1_hp:5.1f}% | E1: {e1_hp:5.1f}% | "
        f"Legal: {legal_count:02d}/18 | "
        f"Rew: {reward:+6.1f}"
    )

    _write_count += 1
    if _write_count % 200 == 0 and not ENV_TAG:   # size check now and then; workers skip it (several processes rotating one
                                                  # file would race), prepare_logs() rotates oversized files at start-up
        _rotate_if_big(LOG_FILE, BIG_TXT_BYTES)
        if _rotate_if_big(CSV_FILE, BIG_CSV_BYTES):
            init_csv()                     # the rotated file is gone: start a fresh one with its header
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")

    # 2. Structured CSV entry for pandas/seaborn post-training analysis
    if not os.path.exists(CSV_FILE):
        init_csv()
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, episode, step, action, action_desc,
            round(p1_hp, 1), round(e1_hp, 1), round(p2_hp, 1), round(e3_hp, 1),
            round(eff, 2), round(action_prob, 3), legal_count,
            round(reward, 2), terminated, player_lost, turn_timeout,
            p1_species, p1_level,
            e1_species, e1_level, e1_status, e1_atk_buff, e1_used_move,
            obs_json
        ])

# gBattleOutcome: 3 is a draw, 4 the player ran, 5 the player's Teleport, 6 the wild mon fled, 8 no Safari Balls left,
# 10 the wild mon teleported.
OUTCOME_NAMES = {1: "WIN", 2: "LOSS", 3: "DRAW", 4: "RAN", 5: "TELEPORT", 6: "FLED", 7: "CAUGHT",
                 8: "NO_SAFARI_BALLS", 9: "FORFEIT", 10: "ENEMY_TELEPORT"}

def outcome_name(outcome):
    return OUTCOME_NAMES.get(outcome, f"END({outcome})")

def log_battle_end(episode, outcome, total_steps, episode_reward, stats=None):
    """outcome is the raw gBattleOutcome. Only a loss/forfeit sounds the alert;
    a catch, a run or a draw is not a loss."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = outcome_name(outcome)
    extra = ""
    if stats:
        kind = {0: "wild", 1: "trainer"}.get(stats.get("trainer_battle"), "?")
        extra = (f" | {kind} L{stats.get('lead_level')} vs L{stats.get('enemy_level')} "
                 f"(gap {stats.get('level_gap'):+d})")
    summary_line = (
        f"[{timestamp}] === Battle #{episode:03d} Result: {result}{extra} | "
        f"Steps: {total_steps} | Total Reward: {episode_reward:+.2f} ==="
    )
    with open(LOG_FILE, "a") as f:
        f.write(summary_line + "\n" + "-"*80 + "\n")

    if outcome in (2, 9):
        alert_loss()


RUNS_FILE = os.path.join(_LOG_DIR, "runs.csv")
RUNS_HEADER = ["timestamp", "run", "script", "obs_dim", "n_actions", "model_loaded", "git", "flags"]
MOVES_FILE = os.path.join(_LOG_DIR, "moves.csv")
MOVES_HEADER = ["timestamp", "run", "episode", "slot", "species", "level", "out_move", "in_move",
                "delta_score", "source", "reply"]


def _git_short():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def log_run_start(script, obs_dim, n_actions, model_loaded=""):
    """One row per process start: which code and settings produced the rows that follow (they carry RUN_ID)."""
    flags = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("POKE_"))
    _append_row(RUNS_FILE, RUNS_HEADER, [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), RUN_ID, script,
                                         obs_dim, n_actions, model_loaded, _git_short(), flags])


PROFILE_FILE = os.path.join(_LOG_DIR, "profile.csv")
PROFILE_HEADER = ["timestamp", "run", "episode", "battles", "wall_s", "breakdown_json"]


def log_profile(episode, battles, wall_s, breakdown):
    """One row per timing report: wall seconds for `battles` battles and {category: [calls, seconds]}. The terminal
    table disappears with the terminal; this keeps it, so the share of time spent walking, waiting and in bridge
    commands can be compared across runs (see log_report.py)."""
    _append_row(PROFILE_FILE, PROFILE_HEADER, [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), RUN_ID, episode, battles,
                                              round(wall_s, 1), json.dumps(breakdown, separators=(",", ":"))])


def log_move(episode, slot, species, level, out_move, in_move, delta_score, source, reply):
    """One row per move-teaching attempt (applied, refused or dry run)."""
    _append_row(MOVES_FILE, MOVES_HEADER, [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), RUN_ID, episode, slot,
                                           species, level, out_move, in_move, round(float(delta_score), 1),
                                           source, reply])


def prepare_logs():
    """Create every log file with the current header, moving old-header files aside. Call once from the main process
    BEFORE starting worker processes, so workers only ever append (two workers creating the same file would race)."""
    _rotate_if_big(LOG_FILE, BIG_TXT_BYTES)
    _rotate_if_big(CSV_FILE, BIG_CSV_BYTES)
    init_csv()
    for path, header in ((BATTLES_FILE, BATTLES_HEADER), (SWAP_FILE, SWAP_HEADER), (CATCH_FILE, CATCH_HEADER),
                         (LEVEL_GAP_FILE, LEVEL_GAP_HEADER), (RUNS_FILE, RUNS_HEADER), (MOVES_FILE, MOVES_HEADER), (PROFILE_FILE, PROFILE_HEADER)):
        if os.path.exists(path):
            _rotate_if_old_header(path, header)
