import os
import socket
import time
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from logger import init_csv, log_step, log_battle_end, decode_action

# --- Load Static Game Data ---
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games", "emerald", "game_data")

with open(os.path.join(_DATA_DIR, "emerald_moves.json")) as f:
    _raw_moves = json.load(f)
MOVE_TYPES = {int(k): v["type_id"] for k, v in _raw_moves.items()}
MOVE_POWER = {int(k): v["power"] for k, v in _raw_moves.items()}
MOVE_ACCURACY = {int(k): v["accuracy"] for k, v in _raw_moves.items()}
MOVE_DAMAGE_CLASS = {int(k): v["damage_class_id"] for k, v in _raw_moves.items()}
MOVE_PRIORITY = {int(k): v.get("priority", 0) for k, v in _raw_moves.items()}
MOVE_NAMES = {int(k): v.get("name", f"Move_{k}") for k, v in _raw_moves.items()}

with open(os.path.join(_DATA_DIR, "species_id_to_name.json")) as f:
    SPECIES_ID_TO_NAME = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(_DATA_DIR, "species_data.json")) as f:
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

with open(os.path.join(_DATA_DIR, "type_chart.json")) as f:
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

def _parse_bench_slot(hp, maxhp, species, item, m1, m2, m3, m4, enemy_t1, enemy_t2):
    """Same treatment _parse_battler() gives active battlers, applied to one
    bench slot. No status/buffs/PP -- those don't apply/aren't sent for a
    benched Pokemon. Returns 19 features: hp_ratio, species, item,
    4 move types, 4 move effectiveness (vs current enemy), 4 move power,
    4 move accuracy."""
    m1_t, m2_t, m3_t, m4_t = (MOVE_TYPES.get(m, 17) for m in (m1, m2, m3, m4))
    m1_eff = get_effectiveness(m1_t, enemy_t1, enemy_t2)
    m2_eff = get_effectiveness(m2_t, enemy_t1, enemy_t2)
    m3_eff = get_effectiveness(m3_t, enemy_t1, enemy_t2)
    m4_eff = get_effectiveness(m4_t, enemy_t1, enemy_t2)

    return [
        hp / max(maxhp, 1), species / 412.0, item / 376.0,
        m1_t / 17.0, m2_t / 17.0, m3_t / 17.0, m4_t / 17.0,
        m1_eff / 4.0, m2_eff / 4.0, m3_eff / 4.0, m4_eff / 4.0,
        MOVE_POWER.get(m1, 0) / 250.0, MOVE_POWER.get(m2, 0) / 250.0,
        MOVE_POWER.get(m3, 0) / 250.0, MOVE_POWER.get(m4, 0) / 250.0,
        MOVE_ACCURACY.get(m1, 100) / 100.0, MOVE_ACCURACY.get(m2, 100) / 100.0,
        MOVE_ACCURACY.get(m3, 100) / 100.0, MOVE_ACCURACY.get(m4, 100) / 100.0,
    ]


# --- Bridge protocol (see battle_bridge.lua header) ---
READY_PHASES = {"CHOOSE_ACTION", "FORCED_SWITCH"}
END_PHASES = {"BATTLE_OVER", "NOT_IN_BATTLE"}
OUTCOME_WON, OUTCOME_LOST, OUTCOME_FORFEITED = 1, 2, 9

# --- Reward ---
# Damage is measured as a fraction of each Pokemon's max HP, summed over the
# whole team, so knocking out one full-HP Pokemon is worth DAMAGE_REWARD_SCALE
# at any level. Using team totals (not the active battler's HP) means
# switching itself never changes the reward: the old reward paid out when
# you switched a hurt Pokemon for a healthy one, and charged you when the
# enemy sent out a fresh one.
DAMAGE_REWARD_SCALE = 50.0
WIN_REWARD, LOSS_REWARD = 100.0, -100.0
STALL_PENALTY = 3.0          # per extra consecutive switch (2nd = -3, 3rd = -6, ...)

TURN_TIMEOUT_S = 60.0        # one turn, including animations; a hang past this truncates
POLL_S = 0.05
MAX_CONSECUTIVE_REJECTS = 10


class PokemonBattleEnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=8888, action_wait=2.0, turn_timeout=TURN_TIMEOUT_S):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock_file = self.sock.makefile("r")
        self.action_wait = action_wait          # kept for compatibility; turns are phase-driven now
        self.turn_timeout = turn_timeout

        # 17 Actions: Moves (0-11) + Bench Switches (12-16)
        self.action_space = spaces.Discrete(17)
        # 417 Dimensions: 4 Battlers (188) + Weather (1) + 12 Bench w/ effectiveness (12*19=228)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(417,), dtype=np.float32)

        init_csv()
        self.episode = 0
        self.step_count = 0
        self.global_step = 0
        self.total_reward = 0.0
        self.move_count = 0
        self.switch_count = 0
        self.illegal_count = 0
        self.current_action_prob = 1.0
        # Consecutive-switch stall tracker: switches in a row with no attack in
        # between. A single switch is never taxed; only stalling is.
        self.consecutive_switches = 0
        self.consecutive_rejects = 0

        self.prev_p_team = None
        self.prev_e_team = None
        self.last_party_raw = [0] * 96
        self.last_active_slot = 0
        self.last_awaiting_forced_switch = False

    # ------------------------------------------------------------- protocol
    def _request(self, line):
        self.sock.sendall((line + "\n").encode())
        reply = self.sock_file.readline()
        if not reply:
            raise ConnectionError("battle_bridge.lua closed the connection")
        return reply.strip()

    def _status(self):
        reply = self._request("STATUS")
        parts = reply.split(":")
        if len(parts) != 3 or parts[0] != "STATUS":
            raise RuntimeError(f"Unexpected STATUS reply {reply!r} -- is the new battle_bridge.lua loaded?")
        return parts[1], parts[2]

    def _wait_for_phase(self, phases, timeout):
        """Polls until the bridge reports one of `phases`. Returns (phase, result),
        or (None, result) on timeout. BUSY/WAITING are never in `phases`, so this
        also waits for the bridge's own key sequence to finish."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            phase, result = self._status()
            if phase in phases:
                return phase, result
            if deadline is not None and time.monotonic() > deadline:
                return None, result
            time.sleep(POLL_S)

    # ---------------------------------------------------------------- state
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
        raw = [int(x) for x in self._request("GET_STATE").split(",")]

        b0_raw, b1_raw, b2_raw, b3_raw = raw[0:21], raw[21:42], raw[42:63], raw[63:84]
        # Lua sends 4 trailing values: weather category, activeSlot (always 0:
        # the active Pokemon is reported first), awaitingForcedSwitch, outcome.
        battle_over, player_lost = raw[84], raw[85]
        party_raw = raw[86:-4]
        weather, active_slot, awaiting_forced_switch, outcome = raw[-4], raw[-3], raw[-2], raw[-1]

        self.last_party_raw = party_raw   # raw ints, unscaled -- used by action_masks()
        self.last_active_slot = active_slot
        self.last_awaiting_forced_switch = bool(awaiting_forced_switch)

        b0_features = self._parse_battler(b0_raw, b1_raw[10], b1_raw[11])
        b1_features = self._parse_battler(b1_raw, b0_raw[10], b0_raw[11])
        b2_features = self._parse_battler(b2_raw, b3_raw[10], b3_raw[11])
        b3_features = self._parse_battler(b3_raw, b2_raw[10], b2_raw[11])

        # party_raw = 48 values (own party: 6 slots x 8) + 48 values (enemy party: 6 x 8)
        p_party_raw, e_party_raw = party_raw[0:48], party_raw[48:96]

        p_bench_features, p_team_slots = [], []
        for i in range(6):
            php, pmax, spc, item, m1, m2, m3, m4 = p_party_raw[i * 8:(i + 1) * 8]
            p_bench_features += _parse_bench_slot(php, pmax, spc, item, m1, m2, m3, m4, b1_raw[10], b1_raw[11])
            p_team_slots.append((php, pmax))

        e_bench_features, e_team_slots = [], []
        for i in range(6):
            spc, item, hp, pmax, m1, m2, m3, m4 = e_party_raw[i * 8:(i + 1) * 8]
            e_bench_features += _parse_bench_slot(hp, pmax, spc, item, m1, m2, m3, m4, b0_raw[10], b0_raw[11])
            e_team_slots.append((hp, pmax))

        # Weather is now a category 0-4 (none, rain, sand, sun, hail) from the
        # real gBattleWeather; the old address was a turn counter.
        obs_list = (b0_features + b1_features + b2_features + b3_features
                    + [weather / 4.0] + p_bench_features + e_bench_features)
        obs = np.array(obs_list, dtype=np.float32)

        return {
            "obs": obs,
            "p_hp": b0_raw[0],
            "e_hp": b1_raw[0],
            "battle_over": bool(battle_over),
            "player_lost": bool(player_lost),
            "outcome": outcome,
            "pp_list": [b0_raw[2], b0_raw[3], b0_raw[4], b0_raw[5]],
            "p_status1": b0_raw[13],
            "moves_list": [b0_raw[6], b0_raw[7], b0_raw[8], b0_raw[9]],
            "p_team": sum(hp / mx for hp, mx in p_team_slots if mx > 0),
            "e_team": sum(hp / mx for hp, mx in e_team_slots if mx > 0),
        }

    def action_masks(self):
        if not hasattr(self, 'last_obs'): return np.ones(17, dtype=bool)
        mask = np.zeros(17, dtype=bool)

        # Forced switch (active battler fainted): only switches are legal.
        # Slot 0 is always the active Pokemon, so 12-16 cover everyone else.
        if getattr(self, 'last_awaiting_forced_switch', False):
            for i in range(1, 6):
                if i == self.last_active_slot: continue
                if self.last_party_raw[i * 8] > 0: mask[11 + i] = True
            if not np.any(mask): mask[12] = True  # failsafe: shouldn't happen if party isn't wiped
            return mask

        pp_ratios = self.last_obs[1:5]
        out_of_all_pp = np.sum(pp_ratios) <= 0.0
        valid_moves = [(out_of_all_pp or pp_ratios[i] > 0.0) for i in range(4)]

        e1_hp, p2_hp, e3_hp = self.last_obs[47], self.last_obs[94], self.last_obs[141]

        for i in range(4):
            if valid_moves[i] and e1_hp > 0.0: mask[i] = True         # Target Enemy Left
            if valid_moves[i] and e3_hp > 0.0: mask[4 + i] = True     # Target Enemy Right
            if valid_moves[i] and p2_hp > 0.0: mask[8 + i] = True     # Target Partner

        for i in range(1, 6):
            if self.last_party_raw[i * 8] > 0: mask[11 + i] = True    # Bench switches (slot 0 = active)

        if not np.any(mask): mask[0] = True  # Failsafe
        return mask

    # ------------------------------------------------------------- episodes
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_count = 0
        self.consecutive_switches = 0
        self.consecutive_rejects = 0

        phase, _ = self._status()
        if phase not in READY_PHASES:
            print("Waiting for the next battle...")
        # Blocks until a battle is on and the game is asking for an action.
        self._wait_for_phase(READY_PHASES, timeout=None)

        st = self._get_state()
        self.prev_p_team, self.prev_e_team = st["p_team"], st["e_team"]
        self.last_obs = st["obs"]
        return st["obs"], {}

    def _outcome_bonus(self, outcome):
        if outcome == OUTCOME_WON:
            return WIN_REWARD
        if outcome in (OUTCOME_LOST, OUTCOME_FORFEITED):
            return LOSS_REWARD
        return 0.0   # ran away, caught, etc.

    def step(self, action):
        # The previous step left the game at a decision point, but make sure.
        phase, _ = self._wait_for_phase(READY_PHASES | END_PHASES, self.turn_timeout)
        if phase is None:
            return self._timeout_return(action, self.last_obs)
        if phase in END_PHASES:
            st = self._get_state()
            return st["obs"], 0.0, True, False, {"player_lost": st["player_lost"], "outcome": st["outcome"]}

        before = self._get_state()   # also refreshes the slot->Pokemon mapping in Lua
        prev_e1_pp = [round(before["obs"][48 + i] * 40.0) for i in range(4)]
        prev_e1_moves = [round(before["obs"][52 + i] * 355.0) for i in range(4)]

        is_attack_action = action < 12
        action_desc = decode_action(action, move_names=[MOVE_NAMES.get(m, f"Move_{m}") for m in before["moves_list"]])

        reply = self._request(f"ACTION:{action}")
        if reply != "ACK":
            # Mask and bridge disagree (should be rare). Nothing happened in game.
            self.consecutive_rejects += 1
            print(f"Bridge rejected ACTION:{action} ({reply})")
            truncated = self.consecutive_rejects >= MAX_CONSECUTIVE_REJECTS
            self.last_obs = before["obs"]
            return before["obs"], 0.0, False, truncated, {"rejected": reply}
        self.consecutive_rejects = 0

        # Wait until the game asks for the next decision or the battle ends.
        # BUSY (bridge pressing keys) and WAITING (animations/text) are skipped.
        phase, result = self._wait_for_phase(READY_PHASES | END_PHASES, self.turn_timeout)
        if phase is None:
            return self._timeout_return(action, before["obs"], action_desc)

        st = self._get_state()
        obs = st["obs"]
        terminated = phase in END_PHASES or st["battle_over"]

        # Reward: share of team HP dealt minus share taken (see constants).
        dealt = self.prev_e_team - st["e_team"]
        taken = self.prev_p_team - st["p_team"]
        reward = DAMAGE_REWARD_SCALE * (dealt - taken)
        if terminated:
            reward += self._outcome_bonus(st["outcome"])

        action_failed = (result == "FAILED")
        if not action_failed:
            if is_attack_action:
                self.consecutive_switches = 0
            else:
                self.consecutive_switches += 1
                if self.consecutive_switches >= 2:
                    reward -= STALL_PENALTY * (self.consecutive_switches - 1)

        self.prev_p_team, self.prev_e_team = st["p_team"], st["e_team"]
        self.total_reward += reward
        if is_attack_action: self.move_count += 1
        else: self.switch_count += 1
        self.global_step += 1

        # --- Live feed & logging ---
        p_hp, e_hp, player_lost = st["p_hp"], st["e_hp"], st["player_lost"]
        p1_spc_id = int(round(obs[43] * 412.0))
        p1_name = SPECIES_ID_TO_NAME.get(p1_spc_id, "Unknown")[:8]
        p1_lvl = int(round(obs[35] * 100.0))

        e1_spc_id = int(round(obs[90] * 412.0))
        e1_name = SPECIES_ID_TO_NAME.get(e1_spc_id, "Unknown")[:8]
        e1_lvl = int(round(obs[82] * 100.0))
        e1_status = int(round(obs[83] * 255.0))
        e1_atk_buff = int(round(obs[87] * 12.0)) - 6

        current_e1_pp = [round(obs[48 + i] * 40.0) for i in range(4)]
        e1_used_move = "-"
        for i in range(4):
            if current_e1_pp[i] < prev_e1_pp[i]:
                e1_used_move = MOVE_NAMES.get(prev_e1_moves[i], "Unknown")
                break

        self.last_obs = obs
        eff = (obs[13 + (action % 4)] * 4.0) if is_attack_action else 1.0
        legal = int(np.sum(self.action_masks()))
        if terminated:
            outcome_txt = {OUTCOME_WON: " | WIN"}.get(st["outcome"], " | LOSS" if player_lost else " | END")
        else:
            outcome_txt = ""
        failed_txt = " | ACTION FAILED" if action_failed else ""

        print(f"GLB {self.global_step:05d} | EP {self.episode:03d} ST {self.step_count:02d} | "
              f"P1: {action_desc[:17].ljust(17)} | E1: {e1_used_move[:12].ljust(12)} | "
              f"P1 HP:{int(p_hp):3d} E1 HP:{int(e_hp):3d} [{e1_name.ljust(8)} L{e1_lvl:2d}] | "
              f"Eff:{eff:3.1f}x AtkBuff:{e1_atk_buff:+2d} | Rew:{reward:+6.1f}{outcome_txt}{failed_txt}")

        obs_dict = {
            "p1_hp": obs[0] * 100,
            "e1_hp": obs[47] * 100,
            "p2_hp": obs[94] * 100,
            "e3_hp": obs[141] * 100,
            "eff_mult": eff,
            "p1_species": p1_name,
            "p1_level": p1_lvl,
            "e1_species": e1_name,
            "e1_level": e1_lvl,
            "e1_status": e1_status,
            "e1_atk_buff": e1_atk_buff,
            "e1_used_move": e1_used_move,
        }
        action_meta = {"desc": action_desc, "legal_count": legal, "prob": getattr(self, "current_action_prob", 0.0)}

        log_step(self.episode, self.step_count, action, reward, terminated, player_lost, obs_dict, action_meta, obs=obs)
        if terminated:
            log_battle_end(self.episode, st["outcome"] == OUTCOME_WON, self.step_count, self.total_reward)

        self.step_count += 1
        return obs, reward, terminated, False, {"player_lost": player_lost, "outcome": st["outcome"],
                                                  "action_failed": action_failed}

    def _timeout_return(self, action, obs, action_desc=None):
        """The game didn't reach a decision point within turn_timeout. That is a
        stuck emulator/script, not something the agent did: truncate (not
        terminate) the episode, with no reward."""
        self.global_step += 1
        p1_spc_id = int(round(obs[43] * 412.0))
        timeout_obs_dict = {
            "p1_hp": obs[0] * 100,
            "p1_species": SPECIES_ID_TO_NAME.get(p1_spc_id, "Unknown")[:8],
            "p1_level": int(round(obs[35] * 100.0)),
        }
        timeout_action_meta = {
            "desc": action_desc or f"action {action}",
            "legal_count": int(np.sum(self.action_masks())),
            "prob": getattr(self, "current_action_prob", 0.0),
        }
        print(f"Turn timed out after {self.turn_timeout:.0f}s -- truncating episode")
        log_step(self.episode, self.step_count, action, 0.0, False, False,
                 timeout_obs_dict, timeout_action_meta, obs=obs, turn_timeout=True)
        self.step_count += 1
        return obs, 0.0, False, True, {"player_lost": False, "turn_timeout": True}

    def close(self):
        print(f"\n=== Session summary ===\nTotal reward: {self.total_reward:+.1f}")
        self.sock.close()
