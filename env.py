import socket
import time
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from logger import init_csv, log_step, log_battle_end, decode_action

# --- Load Static Game Data ---
with open("emerald_moves.json") as f:
    _raw_moves = json.load(f)
MOVE_TYPES = {int(k): v["type_id"] for k, v in _raw_moves.items()}
MOVE_POWER = {int(k): v["power"] for k, v in _raw_moves.items()}
MOVE_ACCURACY = {int(k): v["accuracy"] for k, v in _raw_moves.items()}
MOVE_DAMAGE_CLASS = {int(k): v["damage_class_id"] for k, v in _raw_moves.items()}
MOVE_PRIORITY = {int(k): v.get("priority", 0) for k, v in _raw_moves.items()}
MOVE_NAMES = {int(k): v.get("name", f"Move_{k}") for k, v in _raw_moves.items()}

with open("species_id_to_name.json") as f:
    SPECIES_ID_TO_NAME = {int(k): v for k, v in json.load(f).items()}
with open("species_data.json") as f:
    SPECIES_DATA = json.load(f)

TYPE_NAME_TO_ID = {
    "NORMAL": 0, "FIGHTING": 1, "FLYING": 2, "POISON": 3, "GROUND": 4,
    "ROCK": 5, "BUG": 6, "GHOST": 7, "STEEL": 8, "MYSTERY": 9,
    "FIRE": 10, "WATER": 11, "GRASS": 12, "ELECTRIC": 13, "PSYCHIC": 14,
    "ICE": 15, "DRAGON": 16, "DARK": 17,
}

# Bitmask Status Definitions
STATUS1_SLEEP_MASK = 0x07
STATUS1_PARALYSIS = 0x40
STATUS1_FREEZE = 0x20
STATUS2_CONFUSION_MASK = 0x00000007
STATUS2_SUBSTITUTE     = 0x01000000
STATUS2_TORMENT        = 0x80000000

def get_species_types(species_id):
    name = SPECIES_ID_TO_NAME.get(species_id)
    if not name:
        return 0, 0
    d = SPECIES_DATA.get(name, {"type1": "NORMAL", "type2": "NORMAL"})
    return TYPE_NAME_TO_ID.get(d["type1"], 0), TYPE_NAME_TO_ID.get(d["type2"], 0)

def get_species_info(species_id):
    name = SPECIES_ID_TO_NAME.get(species_id)
    if not name:
        return 0, 0, 300
    d = SPECIES_DATA.get(name, {"type1": "NORMAL", "type2": "NORMAL", "base_total": 300})
    return TYPE_NAME_TO_ID.get(d["type1"], 0), TYPE_NAME_TO_ID.get(d["type2"], 0), d.get("base_total", 300)

def _remap_pokeapi_type(t):
    if t <= 8: return t
    if t == 17: return 9
    return t + 1

with open("type_chart.json") as f:
    _raw_type_chart = json.load(f)
TYPE_CHART = {
    _remap_pokeapi_type(int(atk)): {
        _remap_pokeapi_type(int(dfd)): mult for dfd, mult in defenders.items()
    }
    for atk, defenders in _raw_type_chart.items()
}
TYPE_CHART.setdefault(7, {})[8] = 0.5
TYPE_CHART.setdefault(17, {})[8] = 0.5

def get_effectiveness(move_type, defender_type1, defender_type2):
    mult1 = TYPE_CHART.get(move_type, {}).get(defender_type1, 1.0)
    if defender_type2 != defender_type1:
        mult2 = TYPE_CHART.get(move_type, {}).get(defender_type2, 1.0)
    else:
        mult2 = 1.0
    return mult1 * mult2


class PokemonBattleEnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=8888, action_wait=2.0):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock_file = self.sock.makefile("r")
        self.action_wait = action_wait

        # 17 Actions: Moves (0-11) + Bench Switches (12-16)
        self.action_space = spaces.Discrete(17)
        # 285 Dimensions: 4 Battlers (188) + 12 Bench (96) + 1 Weather
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(285,), dtype=np.float32)

        self.prev_player_hp = None
        self.prev_enemy_hp = None
        init_csv()
        self.episode = 0
        self.step_count = 0
        self.global_step = 0
        self.total_reward = 0.0
        self.move_count = 0
        self.switch_count = 0
        self.illegal_count = 0
        self.current_action_prob = 1.0

    def _parse_battler(self, raw, target_type1, target_type2):
        hp, maxhp, pp1, pp2, pp3, pp4, m1, m2, m3, m4, t1, t2, lvl, s1, s2, atk_b, def_b, spd_b, spc, item, ability = raw

        confused = 1.0 if (s2 & STATUS2_CONFUSION_MASK) != 0 else 0.0
        substitute = 1.0 if (s2 & STATUS2_SUBSTITUTE) != 0 else 0.0
        tormented = 1.0 if (s2 & STATUS2_TORMENT) != 0 else 0.0

        _, _, base_total = get_species_info(spc)
        m1_t, m2_t, m3_t, m4_t = (MOVE_TYPES.get(m, 17) for m in (m1, m2, m3, m4))
        m1_eff = get_effectiveness(m1_t, target_type1, target_type2)
        m2_eff = get_effectiveness(m2_t, target_type1, target_type2)
        m3_eff = get_effectiveness(m3_t, target_type1, target_type2)
        m4_eff = get_effectiveness(m4_t, target_type1, target_type2)

        return [
            hp / max(maxhp, 1), pp1 / 40.0, pp2 / 40.0, pp3 / 40.0, pp4 / 40.0,
            m1 / 355.0, m2 / 355.0, m3 / 355.0, m4 / 355.0,
            m1_t / 17.0, m2_t / 17.0, m3_t / 17.0, m4_t / 17.0,
            m1_eff / 4.0, m2_eff / 4.0, m3_eff / 4.0, m4_eff / 4.0,
            MOVE_POWER.get(m1, 0) / 250.0, MOVE_POWER.get(m2, 0) / 250.0,
            MOVE_POWER.get(m3, 0) / 250.0, MOVE_POWER.get(m4, 0) / 250.0,
            MOVE_ACCURACY.get(m1, 100) / 100.0, MOVE_ACCURACY.get(m2, 100) / 100.0,
            MOVE_ACCURACY.get(m3, 100) / 100.0, MOVE_ACCURACY.get(m4, 100) / 100.0,
            MOVE_DAMAGE_CLASS.get(m1, 0) / 2.0, MOVE_DAMAGE_CLASS.get(m2, 0) / 2.0,
            MOVE_DAMAGE_CLASS.get(m3, 0) / 2.0, MOVE_DAMAGE_CLASS.get(m4, 0) / 2.0,
            MOVE_PRIORITY.get(m1, 0) / 5.0, MOVE_PRIORITY.get(m2, 0) / 5.0,
            MOVE_PRIORITY.get(m3, 0) / 5.0, MOVE_PRIORITY.get(m4, 0) / 5.0,
            t1 / 18.0, t2 / 18.0, lvl / 100.0, s1 / 255.0,
            confused, substitute, tormented,
            atk_b / 12.0, def_b / 12.0, spd_b / 12.0,
            spc / 412.0, base_total / 720.0, item / 376.0, ability / 78.0
        ]

    def _get_state(self):
        self.sock.sendall(b"GET_STATE\n")
        raw = [int(x) for x in self.sock_file.readline().strip().split(",")]

        b0_raw, b1_raw, b2_raw, b3_raw = raw[0:21], raw[21:42], raw[42:63], raw[63:84]
        battle_over, player_lost, party_raw, weather = raw[84], raw[85], raw[86:-1], raw[-1]

        b0_features = self._parse_battler(b0_raw, b1_raw[10], b1_raw[11])
        b1_features = self._parse_battler(b1_raw, b0_raw[10], b0_raw[11])
        b2_features = self._parse_battler(b2_raw, b3_raw[10], b3_raw[11])
        b3_features = self._parse_battler(b3_raw, b2_raw[10], b2_raw[11])

        obs_list = b0_features + b1_features + b2_features + b3_features + [x / 412.0 for x in party_raw] + [weather / 64.0]
        obs = np.array(obs_list, dtype=np.float32)

        pp_list = [b0_raw[2], b0_raw[3], b0_raw[4], b0_raw[5]]
        moves_list = [b0_raw[6], b0_raw[7], b0_raw[8], b0_raw[9]]
        
        return obs, b0_raw[0], b1_raw[0], bool(battle_over), bool(player_lost), pp_list, b0_raw[13], moves_list

    def action_masks(self):
        if not hasattr(self, 'last_obs'): return np.ones(17, dtype=bool)
        mask = np.zeros(17, dtype=bool)

        pp_ratios = self.last_obs[1:5]
        out_of_all_pp = np.sum(pp_ratios) <= 0.0
        valid_moves = [(out_of_all_pp or pp_ratios[i] > 0.0) for i in range(4)]

        e1_hp, p2_hp, e3_hp = self.last_obs[47], self.last_obs[94], self.last_obs[141]

        for i in range(4):
            if valid_moves[i] and e1_hp > 0.0: mask[i] = True         # Target Enemy Left
            if valid_moves[i] and e3_hp > 0.0: mask[4 + i] = True     # Target Enemy Right
            if valid_moves[i] and p2_hp > 0.0: mask[8 + i] = True     # Target Partner

        for i in range(1, 6):
            if self.last_obs[188 + (i * 8)] > 0.0: mask[11 + i] = True # Bench Switches

        if not np.any(mask): mask[0] = True # Failsafe
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_count = 0
        while True:
            obs, p_hp, e_hp, _, _, _, _, _ = self._get_state()
            if e_hp > 0 and p_hp > 0: break
            time.sleep(0.5)

        self.prev_player_hp, self.prev_enemy_hp, self.last_obs = p_hp, e_hp, obs
        return obs, {}

    def step(self, action):
        obs, p_hp, e_hp, battle_over, player_lost, pp_list, p_status1, moves_list = self._get_state()

        # --- Track Enemy PP before the turn resolves ---
        prev_e1_pp = [round(self.last_obs[48 + i] * 40.0) for i in range(4)]
        prev_e1_moves = [round(self.last_obs[52 + i] * 355.0) for i in range(4)]

        if e_hp <= 0 or p_hp <= 0:
            return obs, 0, True, False, {"player_lost": player_lost}

        is_attack_action = action < 12
        action_desc = decode_action(action, move_names=[MOVE_NAMES.get(m, f"Move_{m}") for m in moves_list])

        # --- Execute Keystrokes via Lua Bridge ---
        turn_resolved = False
        initial_pp_sum, initial_p_hp, initial_e_hp = sum(pp_list), p_hp, e_hp

        for _ in range(5):
            if turn_resolved: break
            self.sock.sendall(f"ACTION:{action}\n".encode())
            _ = self.sock_file.readline()

            for _ in range(60):
                time.sleep(0.1)
                (obs, current_p_hp, current_e_hp, current_battle_over, 
                 current_player_lost, current_pp_list, current_p_status1, moves_list) = self._get_state()

                if (sum(current_pp_list) < initial_pp_sum or current_p_hp != initial_p_hp 
                        or current_e_hp != initial_e_hp or current_battle_over):
                    turn_resolved = True
                    break

        # Handle turn timeouts (e.g. paralysis/sleep blocks animations)
        if not turn_resolved:
            if is_attack_action and (current_p_status1 & (STATUS1_PARALYSIS | STATUS1_SLEEP_MASK | STATUS1_FREEZE)):
                self.step_count += 1
                return obs, 0.0, False, False, {"player_lost": False}
            self.total_reward -= 20.0
            self.step_count += 1
            return obs, -20.0, True, False, {"player_lost": False}

        p_hp, e_hp, battle_over, player_lost = current_p_hp, current_e_hp, current_battle_over, current_player_lost

        # Reward Calculation
        reward = float((self.prev_enemy_hp - e_hp) - (self.prev_player_hp - p_hp))
        if battle_over:
            reward += -100.0 if player_lost else 100.0

        self.prev_player_hp, self.prev_enemy_hp = p_hp, e_hp
        self.total_reward += reward
        if is_attack_action: self.move_count += 1
        else: self.switch_count += 1

        # --- NEW LIVE FEED & ENEMY LOGGING BLOCK ---
        self.global_step += 1
        
        # 1. Reverse-engineer Enemy Details
        e1_spc_id = int(obs[90] * 412.0)
        e1_name = SPECIES_ID_TO_NAME.get(e1_spc_id, "Unknown")[:8]
        e1_lvl = int(obs[82] * 100.0)
        e1_status = int(obs[83] * 255.0)
        e1_atk_buff = int(obs[87] * 12.0) - 6
        
        # Detect Enemy Attack by checking if any PP dropped
        current_e1_pp = [round(obs[48 + i] * 40.0) for i in range(4)]
        e1_used_move = "-"
        for i in range(4):
            if current_e1_pp[i] < prev_e1_pp[i]:
                e1_used_move = MOVE_NAMES.get(prev_e1_moves[i], "Unknown")
                break
        
        eff = (obs[13 + (action % 4)] * 4.0) if is_attack_action else 1.0
        legal = int(np.sum(self.action_masks()))
        outcome = " | WIN" if battle_over and not player_lost else " | LOSS" if battle_over else ""

        # 2. Print Enhanced Live Feed (Now showing P1 action vs E1 action)
        print(f"GLB {self.global_step:05d} | EP {self.episode:03d} ST {self.step_count:02d} | "
              f"P1: {action_desc[:17].ljust(17)} | E1: {e1_used_move[:12].ljust(12)} | "
              f"P1 HP:{int(p_hp):3d} E1 HP:{int(e_hp):3d} [{e1_name.ljust(8)} L{e1_lvl:2d}] | "
              f"Eff:{eff:3.1f}x AtkBuff:{e1_atk_buff:+2d} | Rew:{reward:+6.1f}{outcome}")

        # 3. Push Enemy Details to CSV Logger
        obs_dict = {
            "p1_hp": obs[0]*100, 
            "e1_hp": obs[47]*100, 
            "p2_hp": obs[94]*100, 
            "e3_hp": obs[141]*100, 
            "eff_mult": eff,
            "e1_species": e1_name,
            "e1_level": e1_lvl,
            "e1_status": e1_status,
            "e1_atk_buff": e1_atk_buff,
            "e1_used_move": e1_used_move
        }
        
        action_meta = {"desc": action_desc, "legal_count": legal, "prob": getattr(self, "current_action_prob", 0.0)}
        
        log_step(self.episode, self.step_count, action, reward, battle_over, player_lost, obs_dict, action_meta)
        if battle_over: log_battle_end(self.episode, not player_lost, self.step_count, self.total_reward)

        self.step_count += 1
        self.last_obs = obs
        return obs, reward, battle_over, False, {"player_lost": player_lost}

    def close(self):
        print(f"\n=== Session summary ===\nTotal reward: {self.total_reward:+.1f}")
        self.sock.close()
