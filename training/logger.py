import csv
import os
import json
import subprocess
from datetime import datetime

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(_LOG_DIR, "battle_log.txt")
CSV_FILE = os.path.join(_LOG_DIR, "battle_data.csv")

# Predefined action names mapping for 17 discrete actions
TARGET_NAMES = ["Enemy Left", "Enemy Right", "Partner"]

def decode_action(action_idx, move_names=None, party_names=None):
    """
    Decodes discrete action 0-16 into a human-readable action description.
    move_names: list of up to 4 move strings for P1 [m1, m2, m3, m4]
    party_names: list of bench Pokemon names for slots 1-5
    """
    if action_idx < 12:
        move_slot = action_idx % 4
        target_slot = action_idx // 4
        m_name = move_names[move_slot] if move_names and move_slot < len(move_names) else f"Move_{move_slot+1}"
        t_name = TARGET_NAMES[target_slot] if target_slot < len(TARGET_NAMES) else f"Target_{target_slot}"
        return f"{m_name} -> {t_name}"
    else:
        bench_slot = action_idx - 11
        p_name = party_names[bench_slot - 1] if party_names and (bench_slot - 1) < len(party_names) else f"Slot_{bench_slot}"
        return f"SWITCH -> {p_name}"

REJECT_FILE = os.path.join(_LOG_DIR, "rejects.csv")

def log_reject(episode, step, action, reply, phase, mask, retry, substituted_with=None):
    """One row per refused ACTION so the missing mask rule can be found.
    reply is the bridge's REJECT:<reason>; phase is the bridge phase at the
    time; mask is the 17-bool legal-action mask the policy was given."""
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

def log_level_gap(episode, party_levels, box_top, balls, money):
    """Periodic roster snapshot: party levels vs the strongest PC mons."""
    new = not os.path.exists(LEVEL_GAP_FILE)
    with open(LEVEL_GAP_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "episode", "party_levels", "box_top5_levels", "balls", "money"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), episode,
                    " ".join(map(str, party_levels)), " ".join(map(str, box_top)), balls, money])

CATCH_FILE = os.path.join(_LOG_DIR, "catches.csv")

def log_catch(episode, species, level, party_min_level, area_max, balls_used, reason):
    """One row per caught Pokemon. reason is the scripted rule's reason, or
    empty for a catch the rule did not start (a manual throw)."""
    new = not os.path.exists(CATCH_FILE)
    with open(CATCH_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "episode", "species", "level", "party_min_level",
                        "area_max", "balls_used", "reason"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), episode, species, level,
                    party_min_level, area_max, balls_used, reason])

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
    "balls_thrown", "caught",
]
THROWS_FILE = os.path.join(_LOG_DIR, "throws.csv")
THROWS_HEADER = [
    "timestamp", "episode", "map", "species", "level", "ball_number", "result",
    "area_max", "party_min_level", "reason",
]


def _append_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


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
    legal_count = action_meta.get("legal_count", 17)

    # Serialize the full observation (if provided) for post-hoc policy
    # distillation (see distill_policy.py). Kept as its own column so
    # every other column/row shape stays unchanged for existing readers.
    if obs is not None:
        try:
            obs_list = obs.tolist() if hasattr(obs, "tolist") else list(obs)
            obs_json = json.dumps(obs_list)
        except Exception:
            obs_json = ""
    else:
        obs_json = ""

    # 1. Clean, human-readable terminal/log file string
    log_line = (
        f"[{timestamp}] Ep {episode:03d} | Step {step:02d} | "
        f"{action_desc.ljust(26)} | "
        f"Prob: {action_prob*100:5.1f}% | "
        f"Eff: {eff:3.1f}x | "
        f"P1: {p1_hp:5.1f}% | E1: {e1_hp:5.1f}% | "
        f"Legal: {legal_count:02d}/17 | "
        f"Rew: {reward:+6.1f}"
    )

    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")

    # 2. Structured CSV entry for pandas/seaborn post-training analysis
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

OUTCOME_NAMES = {1: "WIN", 2: "LOSS", 3: "RAN", 7: "CAUGHT", 9: "FORFEIT"}

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
