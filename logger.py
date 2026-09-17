import csv 
import os 
import json 
import subprocess 
from datetime import datetime 

LOG_FILE = "battle_log.txt"
CSV_FILE = "battle_data.csv"

def alert_loss():
    print("[ALERT] Battle lost!")
    try:
        subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"])
    except FileNotFoundError:
        os.system('echo -e "\\a"')

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "episode", "step", "action", "obs_json", "reward", "terminated", "player_lost"])

def log_step(episode, step, action, obs, reward, terminated, player_lost):
    timestamp = datetime.now().isoformat()

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, episode, step, action, json.dumps(obs.tolist()), reward, terminated, player_lost
        ])
    
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] ep={episode} step={step} action={action} obs={obs} reward={reward} terminated={terminated}\n")

def log_battle_end(episode, won):
    timestamp = datetime.now().isoformat()
    result = "WIN" if won else "LOSS"
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] === Battle #{episode} ended: {result} ===\n")
    if not won:
        alert_loss()
