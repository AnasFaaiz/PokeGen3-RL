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

def alert_loss():
    print("[ALERT] Battle lost!")
    try:
        subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/message.oga"], check=False)
    except FileNotFoundError:
        os.system('echo -e "\\a"')

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "episode", "step", "action_id", "action_desc",
                "p1_hp_pct", "e1_hp_pct", "p2_hp_pct", "e3_hp_pct",
                "eff_mult", "chosen_prob", "num_legal_actions",
                "reward", "terminated", "player_lost", "turn_timeout",
                "p1_species", "p1_level",
                "e1_species", "e1_level", "e1_status", "e1_atk_buff", "e1_used_move",
                "obs_json"
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

def log_battle_end(episode, won, total_steps, episode_reward):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = "WIN" if won else "LOSS"
    summary_line = (
        f"[{timestamp}] === Battle #{episode:03d} Result: {result} | "
        f"Steps: {total_steps} | Total Reward: {episode_reward:+.2f} ==="
    )
    with open(LOG_FILE, "a") as f:
        f.write(summary_line + "\n" + "-"*80 + "\n")

    if not won:
        alert_loss()
