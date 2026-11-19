import os
import socket
import time
import json
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from catch_policy import CatchContext, CatchPolicy, is_owned, parse_owned
from exp_tables import box_level
from mon_stats import parse_boxmon, compute_stats
import rom_info
from swap_policy import SwapPolicy, BoxEntry
from logger import init_csv, log_step, log_battle_end, log_reject, log_level_gap, log_catch, log_battle_row, log_throw, log_swap, decode_action, outcome_name

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
STATUS2_WRAPPED            = 0x0000E000   # Wrap / Bind / Fire Spin / Clamp ...
STATUS2_ESCAPE_PREVENTION  = 0x04000000   # Mean Look / Spider Web / Block
ABILITY_SHADOW_TAG, ABILITY_MAGNET_PULL, ABILITY_ARENA_TRAP, ABILITY_LEVITATE = 23, 42, 71, 26
TYPE_FLYING, TYPE_STEEL = 2, 8


def is_trapped(player_raw, enemy_raw):
    """True if the game will refuse to switch the player's active Pokemon out.
    Raw battler layout: [10]=type1 [11]=type2 [14]=status2 [20]=ability.
    Anything not visible here (e.g. Ingrain) is caught by the failed-switch
    flag in step()."""
    if player_raw[14] & (STATUS2_WRAPPED | STATUS2_ESCAPE_PREVENTION):
        return True
    e_ability = enemy_raw[20]
    p_types = (player_raw[10], player_raw[11])
    if e_ability == ABILITY_SHADOW_TAG:
        return True
    if e_ability == ABILITY_ARENA_TRAP:
        return TYPE_FLYING not in p_types and player_raw[20] != ABILITY_LEVITATE
    if e_ability == ABILITY_MAGNET_PULL:
        return TYPE_STEEL in p_types
    return False

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
#
# On top of the damage term:
#  * SWITCH_COST   -- flat price for every *voluntary* switch. Forced switches
#                     (after a faint) and switches the game refused are free,
#                     because the agent had no real choice there. A flat cost
#                     (not "only after N in a row") also taxes the
#                     switch/attack/switch/attack stalling pattern. The cost
#                     escalates within a battle (3, 4.5, 6, 7.5) and bench
#                     switches are masked after MAX_VOLUNTARY_SWITCHES, so a
#                     streak gets expensive and bounded. Always a penalty.
#  * TURN_COST     -- price per decision turn so stalling of any kind
#                     (switching, useless moves) is never free.
#  * KO_BONUS      -- extra credit for fainting an enemy / penalty for losing
#                     a Pokemon, on top of the HP-fraction damage.
# Win/loss stays dominant (+-100): a whole fight's shaping is only ~10-30.
DAMAGE_REWARD_SCALE = 50.0
WIN_REWARD, LOSS_REWARD = 100.0, -100.0
SWITCH_COST = 3.0            # base cost of a voluntary switch (the 1st in a battle)
SWITCH_ESCALATION = 0.5      # each further switch in the same battle costs +50% of the base more
MAX_VOLUNTARY_SWITCHES = 4   # hard cap per battle (bench switches are masked after this)
TURN_COST = 1.0              # per decision turn (a 1-turn kill beats a 3-turn kill)
WIN_HP_BONUS = 20.0          # extra on a WIN, scaled by the share of team HP kept vs battle start
KO_BONUS = 5.0               # per enemy faint (+) / own faint (-)

# Catching (outcome 7). Scripted catches happen in reset(), before the agent
# acts, so this only pays for a Pokemon caught by hand mid-fight. Kept well
# below WIN_REWARD. BALL_COST is charged per ball used during the fight.
OUTCOME_CAUGHT = 7
CATCH_BONUS = 10.0
NEW_SPECIES_BONUS = 15.0     # extra when the caught species was not in the Pokedex yet
# What a catch is worth. A useful catch (a training candidate: at least CANDIDATE_GAP levels
# below the party average) is paid close to a win, otherwise the agent could never prefer a
# throw over a guaranteed win. Once the PC holds STOCK_WANTED such candidates, more are worth
# STOCK_FULL_FACTOR as much, so duplicates stay worth a little but ball spam does not pay.
CATCH_USEFUL = 80.0
CATCH_PLAIN = 10.0
CANDIDATE_GAP = 12
STOCK_WANTED = 6
STOCK_FULL_FACTOR = 0.25
LOW_HP_CATCH_BONUS = 10.0    # catch landed with the enemy under 50% HP (Gen 3 catch rate rises as HP drops)
BALL_COST = 1.0
MAX_BALLS_PER_BATTLE = 4     # throw action is masked after this many balls in one battle

# Spreading XP: the lead outgrows everything while the bench stays low, and wild
# fights stop teaching anything. LEVEL_REWARD pays for growth on a log scale, so a
# level on an L30 Pokemon is worth ~3x a level on an L88 one. A switch to a lower-level
# bench member is free while the lead is SWITCH_FREE_GAP+ levels above the enemy.
LEVEL_REWARD = 100.0
REWARD_SCALE = 0.01          # returns of +-100 -> +-1, so the critic's loss isn't dominated by outcome bonuses
SWITCH_FREE_GAP = 15

THROW_ACTION = 17            # model-chosen "throw a ball" (wild battles only)
N_ACTIONS = 18               # 0-11 moves, 12-16 bench switches, 17 throw
BASE_OBS_DIM = 417
EXTRA_FEATURES = 14          # appended after the original 417 so expand_model-style surgery works
OBS_DIM = BASE_OBS_DIM + EXTRA_FEATURES

TURN_TIMEOUT_S = 60.0        # one turn, including animations; a hang past this truncates
POLL_S = 0.05
MAX_CONSECUTIVE_REJECTS = 10
BOX_CACHE_BATTLES = 10       # refresh the PC snapshot at most this often (and after a catch)
SWAP_STARTUP_FILL = True    # first swap pass after launch brings the whole party down to the area's level band
SCRIPTED_SWAP = True        # exchange an overleveled party mon for a PC mon near the area's level
SCRIPTED_CATCH = False      # catching is now a model action (THROW_ACTION); True re-enables the old script
PROFILE_EVERY = 25           # print a timing table every this many battles (0 = off)
ROSTER_LOG_EVERY = 25        # battles between party-vs-PC level snapshots

# --- Overworld navigation (scripted, never learned) ---
# Between battles the bridge walks the player back to wild grass, and to a
# Pokemon Center first when the party is in no state to fight. None of this
# is in the observation or action space: the model only plays battles.
#
# Healing before a wipe rather than after is worth the detour -- a whiteout
# costs money and drops the player at the last heal point, which may be
# nowhere near the grass the agent was training in.
HEAL_HP_FRACTION = 0.35      # heal when the team is below this share of max HP
HEAL_MIN_CONSCIOUS = 2       # ...or when fewer than this many can still fight
NAV_TIMEOUT_S = 600.0
NAV_POLL_S = 0.25
NAV_RETRIES = 5
NAV_RETRY_DELAY_S = 2.0
NAV_GIVEUP_PAUSE_S = 30.0   # pause between full retry rounds
OVERWORLD_WAIT_S = 60.0


class PokemonBattleEnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=8888, action_wait=2.0, turn_timeout=TURN_TIMEOUT_S):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock_file = self.sock.makefile("r")
        self.action_wait = action_wait          # kept for compatibility; turns are phase-driven now
        self.turn_timeout = turn_timeout

        # 18 Actions: Moves (0-11) + Bench Switches (12-16) + Throw ball (17)
        self.action_space = spaces.Discrete(N_ACTIONS)
        # 429 Dimensions: 4 Battlers (188) + Weather (1) + 12 Bench w/ effectiveness (12*19=228)
        # + 12 appended extras: 6 party levels, level gap, party min/mean level,
        # balls left, wild flag, enemy species already owned
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        init_csv()
        self.episode = 0
        self.step_count = 0
        self.global_step = 0
        self.total_reward = 0.0          # per battle (zeroed in reset)
        self.session_reward = 0.0        # whole run, for the summary
        self.move_count = 0
        self.switch_count = 0            # voluntary switches only
        self.forced_switch_count = 0     # replacements after a faint (not the agent's choice)
        self.failed_switch_count = 0     # switches the game refused
        self.illegal_count = 0
        self.current_action_prob = 1.0
        self.consecutive_rejects = 0

        self.prev_p_team = None
        self.start_p_team = None
        self.prev_e_team = None
        self.prev_p_alive = None
        self.battle_start = {"lead_level": 0, "enemy_level": 0, "trainer": -1, "balls": None}
        self.catch_policy = CatchPolicy(enabled=SCRIPTED_CATCH)
        self.swap_policy = SwapPolicy(enabled=SCRIPTED_SWAP)
        self.box_entries = []            # BoxEntry list matching box_cache
        self.swaps = 0                   # swaps since the previous finished battle
        self.swap_startup_done = False   # the one-time start-up fill has run
        self._prof = {}                  # key -> [count, seconds]; see _prof_report()
        self._prof_reset_t = None        # when the previous reset() began
        self._battle_inv = None          # INVENTORY read at this battle's start, reused once
        self.level_start = []
        self.box_cache = None            # (levels, species set) of the PC, refreshed lazily
        # Scripted throws/catches happen in reset(), before any step(), so they never
        # reach a battle's own stats. Count them here and flush into the next battle_stats.
        self.scripted_balls = 0
        self.scripted_caught = 0
        self._last_skip_reason = None
        self.box_cache_episode = 0
        self.last_roster_episode = -1
        self.prev_e_alive = None
        self.last_party_raw = [0] * 96
        self.last_active_slot = 0
        self.last_awaiting_forced_switch = False
        self.last_trapped = False
        self.switch_blocked = False   # a switch was refused this battle
        self.party_levels = [0] * 6   # reported slot order (active first); refreshed from the bridge
        self.level_sum_start = None   # sum(ln(level)) over the party at battle start
        self.balls_left = 0
        self.balls_thrown_now = 0     # balls thrown by the agent in this battle
        self.enemy_species = 0
        self.enemy_owned = None       # Pokedex says the enemy species is already caught
        self.is_wild = False
        self.box_low_stock = 0        # PC Pokemon that are low enough to be training candidates
        self.last_throw_e_hp = 1.0    # enemy HP share when the last ball was thrown

    # ------------------------------------------------------------- protocol
    def _request(self, line):
        t0 = time.perf_counter()
        self.sock.sendall((line + "\n").encode())
        reply = self.sock_file.readline()
        if not reply:
            raise ConnectionError("battle_bridge.lua closed the connection")
        if PROFILE_EVERY:
            self._prof_add("cmd:" + line.split(":")[0].rstrip("0123456789"), time.perf_counter() - t0)
        return reply.strip()

    def _prof_add(self, key, dt):
        rec = self._prof.setdefault(key, [0, 0.0])
        rec[0] += 1
        rec[1] += dt

    def _prof_report(self):
        """Where wall-clock time went since the last report: bridge commands, overworld
        navigation, and the rest (battle turns, animations, policy inference)."""
        wall = self._prof.get("episode_wall", [0, 0.0])[1] or 1e-9
        n = max(self._prof.get("episode_wall", [1, 0])[0], 1)
        print(f"[profile] {n} battles, {wall:.0f}s wall ({wall / n:.1f}s per battle)")
        rows = sorted(((k, c, t) for k, (c, t) in self._prof.items() if k != "episode_wall"),
                      key=lambda r: -r[2])
        for k, c, t in rows[:12]:
            print(f"[profile]   {k:<22} {c:>6} calls  {t:>7.1f}s  {100 * t / wall:>5.1f}%  "
                  f"{1000 * t / max(c, 1):>7.1f} ms/call")
        self._prof = {}

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

    def _extra_features(self, lead_level, enemy_level):
        """The 12 values appended to the observation: what the agent needs to
        judge training value (bench levels, level gap) and whether a throw is
        worthwhile (balls, wild flag, species already owned)."""
        lv = [min(max(int(x), 0), 100) for x in self.party_levels[:6]]
        lv += [0] * (6 - len(lv))
        alive_lv = [x for x in lv if x > 0]
        gap = max(0, lead_level - enemy_level)
        owned = 0.5 if self.enemy_owned is None else float(self.enemy_owned)
        return [x / 100.0 for x in lv] + [
            min(gap, 100) / 100.0,
            (min(alive_lv) / 100.0) if alive_lv else 0.0,
            (sum(alive_lv) / len(alive_lv) / 100.0) if alive_lv else 0.0,
            min(self.balls_left, 50) / 50.0,
            1.0 if self.is_wild else 0.0,
            owned,
            min(self.box_low_stock, STOCK_WANTED) / STOCK_WANTED,
            min(max(((sum(alive_lv) / len(alive_lv)) if alive_lv else 0) - enemy_level, 0), 100) / 100.0,
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
        self.last_trapped = is_trapped(b0_raw, b1_raw)

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
        # Party levels in the same (active-first) slot order, appended by the newer
        # battle_bridge.lua after the enemy party. An older bridge sends none: keep
        # whatever the battle-start INVENTORY read gave us.
        if len(party_raw) >= 102:
            self.party_levels = [int(x) for x in party_raw[96:102]]
        extras = self._extra_features(b0_raw[12], b1_raw[12])

        obs_list = (b0_features + b1_features + b2_features + b3_features
                    + [weather / 4.0] + p_bench_features + e_bench_features + extras)
        obs = np.array(obs_list, dtype=np.float32)

        return {
            "obs": obs,
            "e_species": b1_raw[18],
            "e_level": b1_raw[12],
            "p_level": b0_raw[12],
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
            "p_alive": sum(1 for hp, mx in p_team_slots if mx > 0 and hp > 0),
            "e_alive": sum(1 for hp, mx in e_team_slots if mx > 0 and hp > 0),
        }

    def action_masks(self, refresh=True):
        # Rebuild the cached state from the live game first. The mask used to
        # come from the observation taken at the end of the previous step, so
        # anything that changed since (forced switch, party reorder, manual
        # input) made it disagree with what the bridge would accept.
        if refresh and hasattr(self, 'last_obs'):
            try:
                phase, _ = self._status()
                if phase in READY_PHASES:
                    self.last_obs = self._get_state()["obs"]
            except (OSError, ValueError, RuntimeError):
                pass   # keep the cached state
        if not hasattr(self, 'last_obs'): return np.ones(N_ACTIONS, dtype=bool)
        mask = np.zeros(N_ACTIONS, dtype=bool)

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

        # A damaging move the enemy is immune to (0.0x) only wastes the turn.
        # Status moves can still hit, and Hidden Power's type is not known here.
        for i in range(4):
            move_id = int(round(self.last_obs[5 + i] * 355.0))
            is_damaging = MOVE_DAMAGE_CLASS.get(move_id, 0) != 0
            if is_damaging and move_id != 237 and self.last_obs[13 + i] <= 0.0:
                valid_moves[i] = False
        if not any(valid_moves):    # never leave the agent without a move
            valid_moves = [(out_of_all_pp or pp_ratios[i] > 0.0) for i in range(4)]

        e1_hp, p2_hp, e3_hp = self.last_obs[47], self.last_obs[94], self.last_obs[141]

        for i in range(4):
            if valid_moves[i] and e1_hp > 0.0: mask[i] = True         # Target Enemy Left
            if valid_moves[i] and e3_hp > 0.0: mask[4 + i] = True     # Target Enemy Right
            if valid_moves[i] and p2_hp > 0.0: mask[8 + i] = True     # Target Partner

        # Trapped (Mean Look, Shadow Tag, Ingrain...): the game refuses the
        # switch and the bridge wastes ~3 s in the party menu, so hide them.
        can_switch = not (getattr(self, 'last_trapped', False) or getattr(self, 'switch_blocked', False))
        if self.switch_count >= MAX_VOLUNTARY_SWITCHES:
            can_switch = False
        for i in range(1, 6):
            if can_switch and self.last_party_raw[i * 8] > 0: mask[11 + i] = True    # Bench switches (slot 0 = active)

        # Throw a ball: wild battles only, balls in the bag, capped per battle.
        if (self.is_wild and self.balls_left > 0 and self.balls_thrown_now < MAX_BALLS_PER_BATTLE
                and e1_hp > 0.0):
            mask[THROW_ACTION] = True

        if not np.any(mask): mask[0] = True  # Failsafe
        return mask

    # ------------------------------------------------------------- episodes
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        reset_t0 = time.perf_counter()
        if PROFILE_EVERY:
            if self._prof_reset_t is not None:
                self._prof_add("episode_wall", reset_t0 - self._prof_reset_t)
            self._prof_reset_t = reset_t0
            if self.episode % PROFILE_EVERY == 0:
                self._prof_report()
        self.step_count = 0
        self.total_reward = 0.0
        self.move_count = self.switch_count = 0          # per-battle stats
        self.forced_switch_count = self.failed_switch_count = 0
        self.switch_blocked = False
        self.last_trapped = False
        self.consecutive_rejects = 0

        while True:
            phase, _ = self._status()
            if phase not in READY_PHASES:
                t_sw = time.perf_counter()
                self._maybe_swap()
                self._prof_add("swap_check", time.perf_counter() - t_sw)
                # Not mid-battle: walk the player to the next encounter, healing
                # first if the party can't take another fight.
                t_nav = time.perf_counter()
                self._navigate_to_next_battle()
                self._prof_add("nav_to_battle", time.perf_counter() - t_nav)

            phase, _ = self._status()
            if phase not in READY_PHASES:
                print("Waiting for the next battle...")
            # Blocks until a battle is on and the game is asking for an action.
            t_wait = time.perf_counter()
            self._wait_for_phase(READY_PHASES, timeout=None)
            self._prof_add("wait_for_battle", time.perf_counter() - t_wait)

            st = self._get_state()
            self._record_battle_start(st["obs"])
            st = self._get_state()   # rebuild so the extra features reflect this battle
            # Scripted catching happens here, before the agent's first move, so
            # no action the policy chose is ever replaced. If the battle ends
            # during it (caught, or it fled), loop to the next encounter.
            ended, st = self._catch_before_fight(st)
            if not ended:
                break
        self.start_p_team = st["p_team"]
        self.prev_p_team, self.prev_e_team = st["p_team"], st["e_team"]
        self.prev_p_alive, self.prev_e_alive = st["p_alive"], st["e_alive"]
        self.last_obs = st["obs"]
        return st["obs"], {}

    def _get_box(self, force=False):
        """(levels, species set) of every PC mon, cached across battles; None if unreadable."""
        stale = self.episode - self.box_cache_episode >= BOX_CACHE_BATTLES
        if self.box_cache is None or stale or force:
            try:
                line = self._request("BOX")
                if not line.startswith("BOX:") or line.startswith("BOX:ERR"):
                    return None
                levels, species, entries = [], set(), []
                parsed = [ent.split(":") for ent in filter(None, line[4:].split(","))]
                rom_info.ensure(self._request, [int(p[1]) for p in parsed])   # ROM growth rates, cached
                for pos, sp, exp in parsed:
                    levels.append(box_level(int(sp), int(exp)))
                    species.add(int(sp))
                    entries.append(BoxEntry(pos, int(sp), levels[-1]))
                self.box_entries = entries
                self.box_cache, self.box_cache_episode = (levels, species), self.episode
            except (OSError, ValueError):
                return None
        return self.box_cache

    def _maybe_swap(self):
        """Between battles: exchange every overleveled party mon (any slot) for a PC mon near
        the area's level, in one pass. The first pass after launch is a start-up fill that
        brings the whole party down to the area's band."""
        pol = self.swap_policy
        if not pol.due(self.episode):
            return
        inv = self._read_inventory()
        if not inv or not inv["lv"]:
            return
        party = [int(x) for x in inv["lv"].split(",") if x]
        map_key = inv.get("map", "?")
        area_max = self.catch_policy.area_max(map_key)
        startup = SWAP_STARTUP_FILL and not self.swap_startup_done
        if not pol.overleveled_slots(party, area_max, startup):
            if startup and area_max is not None:
                self.swap_startup_done = True        # nothing to fill: the party is already in band
            return                                   # cheap gate first: no box read unless it matters
        box = self._get_box()                        # cached; refreshed after a catch or swap
        if not box or not pol.candidates(area_max, self.box_entries):
            pol.next_check_episode = self.episode + pol.retry
            return
        done = 0
        try:
            species = []
            for i in range(len(party)):
                mon = parse_boxmon(self._request(f"PARTYMON:{i}"))
                species.append(mon["species"] if mon else 0)
            picks = pol.plan(self.episode, party, area_max, self.box_entries,
                             party_species=species, startup=startup)
            if not picks:
                return
            for slot, ent in picks:
                mon = parse_boxmon(self._request(f"BOXMON:{ent.key}"))
                if mon:
                    rom_info.ensure(self._request, [mon["species"]])
                stats = compute_stats(mon) if mon else None
                if not stats:
                    print(f"[swap] cannot compute stats for {ent.key}; skipped")
                    continue
                reply = self._request(f"SWAP:{slot}:{ent.key}:" + ":".join(map(str, stats)))
                name = SPECIES_ID_TO_NAME.get(ent.species, ent.species)
                log_swap(self.episode, map_key, area_max, slot, party[slot], name, ent.level, reply)
                print(f"[swap] slot {slot} L{party[slot]} <-> box {ent.key} {name} L{ent.level}: {reply}")
                if reply != "SWAP:OK":
                    break                              # refused (e.g. in battle): stop the pass
                done += 1
        except (OSError, ValueError, KeyError) as e:
            print(f"[swap] failed: {e}")
        if startup:
            self.swap_startup_done = True
        if done:
            self.swaps += done
            self.box_cache = None
            inv = self._read_inventory()
            if inv and inv["lv"]:
                self.party_levels = ([int(x) for x in inv["lv"].split(",") if x] + [0] * 6)[:6]
        else:
            pol.last_swap_episode = -10**9           # nothing swapped: retry after the back-off, not the cooldown
            pol.next_check_episode = self.episode + pol.retry

    def _catch_before_fight(self, st):
        """Run the catch rule at the start of a wild battle. Returns
        (battle_ended, state). The battle ends here if a ball catches the
        Pokemon or the Pokemon flees."""
        pol = self.catch_policy
        # _record_battle_start read INVENTORY moments ago, with no game progress since.
        inv, self._battle_inv = self._battle_inv or self._read_inventory(), None
        if inv and not pol.enabled and inv["trainer"] != "1" and st["e_level"] > 0:
            pol.observe_area(inv.get("map", "?"), st["e_level"])   # the swap rule needs the area's level even when throws are the model's
        if not pol.enabled or not inv:
            return False, st
        trainer = inv["trainer"] == "1"
        map_key = inv.get("map", "?")
        if not trainer and st["e_level"] > 0:
            pol.observe_area(map_key, st["e_level"])
        start_balls = self._ball_total(inv)
        while True:
            party = [int(x) for x in inv["lv"].split(",") if x]
            area_max = pol.area_max(map_key)
            box = self._get_box() if pol.overleveled(party, area_max) else (None, set())
            ctx = CatchContext(
                enemy_species=st["e_species"], enemy_level=st["e_level"], party_levels=party,
                box_levels=box[0] if box else None, box_species=box[1] if box else set(),
                area_max=area_max, trainer=trainer, balls=self._ball_total(inv) or 0)
            if not pol.decide(ctx):
                # Print only when the reason changes, so a long run of identical skips doesn't spam.
                if pol.last_reason != self._last_skip_reason:
                    self._last_skip_reason = pol.last_reason
                    print(f"[catch] skip: {pol.last_reason} (map {map_key}, enemy L{ctx.enemy_level}, "
                          f"party min L{min(party) if party else '?'}, area max {area_max})")
                break
            self._last_skip_reason = None
            sp_name = SPECIES_ID_TO_NAME.get(ctx.enemy_species, ctx.enemy_species)
            reply = self._request("THROW")
            if reply != "ACK":
                print(f"[catch] throw rejected: {reply}")
                pol.record_throw(ok=False)
                log_throw(self.episode, map_key, sp_name, ctx.enemy_level, pol.thrown + 1,
                          f"REJECTED:{reply}", area_max, min(party), pol.last_reason)
                break
            self.scripted_balls += 1
            print(f"[catch] throwing at L{ctx.enemy_level} (ball {pol.thrown + 1}, "
                  f"area max L{area_max}, party min L{min(party)})")
            phase, result = self._wait_for_phase(READY_PHASES | END_PHASES, self.turn_timeout)
            pol.record_throw(ok=(result != "FAILED"))
            if phase is None:
                break
            st = self._get_state()
            ended = phase in END_PHASES or st["battle_over"]
            log_throw(self.episode, map_key, sp_name, ctx.enemy_level, pol.thrown,
                      outcome_name(st["outcome"]) if ended else ("FAILED" if result == "FAILED" else "OK"),
                      area_max, min(party), pol.last_reason)
            if phase in END_PHASES or st["battle_over"]:
                self._finish_catch_battle(st, ctx, start_balls)
                return True, st
            if phase == "FORCED_SWITCH":
                break
            inv = self._read_inventory()
            if not inv:
                break
        # The agent is not charged for HP lost or balls spent during scripted throws.
        self.battle_start["balls"] = self._ball_total(self._read_inventory())
        return False, st

    def _finish_catch_battle(self, st, ctx, start_balls):
        used = None
        end = self._ball_total(self._read_inventory())
        if start_balls is not None and end is not None:
            used = max(0, start_balls - end)
        if st["outcome"] == OUTCOME_CAUGHT:
            self.box_cache = None   # the new mon changes the box
            self.scripted_caught += 1
            log_catch(self.episode, SPECIES_ID_TO_NAME.get(ctx.enemy_species, ctx.enemy_species),
                      ctx.enemy_level, min(ctx.party_levels) if ctx.party_levels else 0,
                      ctx.area_max, used, self.catch_policy.last_reason)
            print(f"[catch] caught L{ctx.enemy_level} {SPECIES_ID_TO_NAME.get(ctx.enemy_species, '?')}")
        else:
            print(f"[catch] battle ended during throws ({outcome_name(st['outcome'])})")

    def _read_inventory(self):
        """Decode the bridge's INVENTORY reply into a dict, or None."""
        try:
            line = self._request("INVENTORY")
            if not line.startswith("INV:") or line.startswith("INV:ERR"):
                return None
            return dict(kv.split("=", 1) for kv in line[4:].split(";"))
        except (OSError, ValueError):
            return None

    def _party_avg(self):
        alive = [x for x in self.party_levels if x > 0]
        return sum(alive) / len(alive) if alive else 0.0

    def _catch_value(self):
        """Reward for a successful catch: big for a training candidate while the PC
        stock is short, small otherwise, plus bonuses for a new species and a low-HP catch."""
        enemy_level = self.battle_start["enemy_level"]
        useful = enemy_level <= self._party_avg() - CANDIDATE_GAP
        value = CATCH_USEFUL if useful else CATCH_PLAIN
        if useful and self.box_low_stock >= STOCK_WANTED:
            value *= STOCK_FULL_FACTOR
        if self.enemy_owned is False:
            value += NEW_SPECIES_BONUS
        elif self.box_cache and self.enemy_species in self.box_cache[1]:
            # A species the PC already holds and the Pokedex has: nothing gained, and balls
            # can't be rebought. (Independent of `useful`: the party average is far above the area.)
            return 0.0
        if self.last_throw_e_hp < 0.5:
            value += LOW_HP_CATCH_BONUS
        return value

    @staticmethod
    def _log_level_sum(levels):
        """Sum of ln(level) over the party. Order-independent, so in-battle party
        reshuffles don't matter, and a level on a low-level mon counts for more."""
        return sum(math.log(max(l, 1)) for l in levels)

    def _record_battle_start(self, obs):
        """Remember the lead/enemy levels this battle began with, and every
        ROSTER_LOG_EVERY battles snapshot party vs PC levels. Logging only:
        nothing here touches the observation or reward."""
        inv = self._read_inventory()
        self._battle_inv = inv
        self.battle_start = {
            "lead_level": int(round(obs[35] * 100.0)),
            "enemy_level": int(round(obs[82] * 100.0)),
            "trainer": int(inv["trainer"]) if inv else -1,
            "balls": self._ball_total(inv),
        }
        self.catch_policy.start_battle()

        # Per-battle state the observation, throw mask and rewards use.
        self.is_wild = bool(inv) and inv["trainer"] == "0"
        self.balls_left = self._ball_total(inv) or 0
        self.balls_thrown_now = 0
        lv = [int(x) for x in inv["lv"].split(",") if x] if inv else []
        self.level_sum_start = self._log_level_sum(lv) if lv else None
        self.level_start = lv
        if lv and not any(self.party_levels):
            self.party_levels = (lv + [0] * 6)[:6]   # older bridge: game order, not reported order
        self.box_low_stock = 0
        if self.is_wild:
            box = self._get_box()
            if box:
                limit = self._party_avg() - CANDIDATE_GAP
                self.box_low_stock = sum(1 for l in box[0] if l <= limit)
        species = int(round(obs[90] * 412.0))
        self.enemy_species = species
        try:
            self.enemy_owned = is_owned(parse_owned(self._request("DEX")), species) if self.is_wild else None
        except (OSError, ValueError):
            self.enemy_owned = None
        if inv and self.episode % ROSTER_LOG_EVERY == 1 and self.episode != self.last_roster_episode:
            self.last_roster_episode = self.episode
            try:
                box = self._request("BOX")
                top = []
                if box.startswith("BOX:") and not box.startswith("BOX:ERR"):
                    lv = []
                    parsed = [ent.split(":") for ent in filter(None, box[4:].split(","))]
                    rom_info.ensure(self._request, [int(p[1]) for p in parsed])
                    for _, sp, exp in parsed:
                        lv.append(box_level(int(sp), int(exp)))
                    top = sorted(lv, reverse=True)[:5]
                log_level_gap(self.episode, inv["lv"].split(",") if inv["lv"] else [],
                              top, inv["balls"], inv["money"])
            except (OSError, ValueError):
                pass

    def _party_hp(self):
        """(current HP, max HP, conscious count) across the party."""
        reply = self._request("PARTY_HP")
        parts = reply.split(":")
        if len(parts) != 4 or parts[0] != "PARTY_HP":
            return None, None, None
        return int(parts[1]), int(parts[2]), int(parts[3])

    def _needs_healing(self):
        cur, mx, conscious = self._party_hp()
        if not mx:
            return False
        if cur / mx < HEAL_HP_FRACTION:
            return True
        # A wipe can also be one battle away while the HP fraction still looks
        # fine, if everything left is concentrated in a single Pokemon.
        return 0 < conscious < HEAL_MIN_CONSCIOUS

    def _wait_for_overworld(self):
        """After a battle the bridge keeps clicking through post-battle screens
        (level-ups, evolution) while the game still reports in-battle, and
        NAV is rejected until that is over. Returns True once the player is
        free in the overworld, False if a new battle is ready first or the
        wait runs out."""
        deadline = time.monotonic() + OVERWORLD_WAIT_S
        while time.monotonic() < deadline:
            phase, _ = self._status()
            if phase == "NOT_IN_BATTLE":
                return True
            if phase in READY_PHASES:
                return False
            time.sleep(POLL_S)
        return False

    def _watch_nav(self):
        """Poll NAV:STATUS until it settles. Returns "battle", "failed",
        "idle" or "timeout"."""
        deadline = time.monotonic() + NAV_TIMEOUT_S
        last_phase = None
        while time.monotonic() < deadline:
            status = self._request("NAV:STATUS")
            # Surface Match Call / text-box handling so it is visible in training logs.
            phase_name = status.split(" ")[0]
            if phase_name != last_phase:
                if phase_name == "NAV:WALKING:dialog":
                    print("[nav] field input locked (text box / call / scene) -- tapping through it")
                elif last_phase == "NAV:WALKING:dialog":
                    print("[nav] dialog cleared -- resuming")
                last_phase = phase_name
            if status.startswith(("NAV:BATTLE", "NAV:ARRIVED")):
                return "battle"
            if status.startswith("NAV:FAILED"):
                print(f"[nav] {status}")
                return "failed"
            if status == "NAV:IDLE":
                return "idle"
            # A battle can start mid-walk; the bridge reports it, but check the
            # battle phase too so we never miss the handoff.
            phase, _ = self._status()
            if phase in READY_PHASES:
                return "battle"
            time.sleep(NAV_POLL_S)
        self._request("NAV:CANCEL")
        return "timeout"

    def _navigate_to_next_battle(self):
        """Walk to encounter tiles (grass or cave; healing first if needed) and block until a battle
        starts. Failures are not fatal: the player may be mid-cutscene (a
        whiteout replays the Center scene), so retry a few times. If every
        attempt fails, pause briefly (a battle may still start on its own) and
        run the whole retry round again, so training never stalls for good."""
        while True:
            for attempt in range(1, NAV_RETRIES + 1):
                if not self._wait_for_overworld():
                    return

                try:
                    goal = "HEAL" if self._needs_healing() else "HUNT"
                except (OSError, ValueError):
                    goal = "HUNT"

                reply = self._request(f"NAV:{goal}")
                if reply != "ACK":
                    print(f"[nav] {goal} rejected: {reply} (attempt {attempt}/{NAV_RETRIES})")
                    time.sleep(NAV_RETRY_DELAY_S)
                    continue

                if goal == "HEAL":
                    print("[nav] party is low -- heading to the Pokemon Center")

                outcome = self._watch_nav()
                if outcome in ("battle", "idle"):
                    return
                if outcome == "timeout":
                    print("[nav] timed out -- continuing without it")
                    return
                print(f"[nav] attempt {attempt}/{NAV_RETRIES} failed")
                time.sleep(NAV_RETRY_DELAY_S)

            print(f"[nav] all {NAV_RETRIES} attempts failed -- pausing {NAV_GIVEUP_PAUSE_S:.0f}s, then retrying")
            deadline = time.monotonic() + NAV_GIVEUP_PAUSE_S
            while time.monotonic() < deadline:
                phase, _ = self._status()
                if phase in READY_PHASES:
                    return
                time.sleep(NAV_POLL_S)

    def _outcome_bonus(self, outcome):
        if outcome == OUTCOME_WON:
            return WIN_REWARD
        if outcome in (OUTCOME_LOST, OUTCOME_FORFEITED):
            return LOSS_REWARD
        if outcome == OUTCOME_CAUGHT:
            return 0.0   # paid by _catch_value, which knows the level and PC stock
        return 0.0   # ran away, etc.

    @staticmethod
    def _ball_total(inv):
        """Total balls in the bag from an INVENTORY dict ("id:qty,id:qty"), or None."""
        if not inv:
            return None
        return sum(int(b.split(":")[1]) for b in inv["balls"].split(",") if b)

    def step(self, action):
        # The previous step left the game at a decision point, but make sure.
        phase, _ = self._wait_for_phase(READY_PHASES | END_PHASES, self.turn_timeout)
        if phase is None:
            return self._timeout_return(action, self.last_obs)
        if phase in END_PHASES:
            st = self._get_state()
            return st["obs"], 0.0, True, False, {"player_lost": st["player_lost"], "outcome": st["outcome"]}

        forced_switch_turn = (phase == "FORCED_SWITCH")   # replacement after a faint: no real choice
        before = self._get_state()   # also refreshes the slot->Pokemon mapping in Lua
        prev_e1_pp = [round(before["obs"][48 + i] * 40.0) for i in range(4)]
        prev_e1_moves = [round(before["obs"][52 + i] * 355.0) for i in range(4)]

        # A rejected ACTION means the mask and the game disagree (state moved
        # under us, e.g. manual keypresses). Never hand that back to SB3 as a
        # transition: it would be stored as a real (reward 0) step. Retry the
        # same action once the state settles, then fall back to a legal one;
        # only give up (truncate) after MAX_CONSECUTIVE_REJECTS.
        reject_retries = 0
        while True:
            is_attack_action = action < 12
            is_throw = (action == THROW_ACTION)
            action_desc = decode_action(action, move_names=[MOVE_NAMES.get(m, f"Move_{m}") for m in before["moves_list"]])
            reply = self._request("THROW" if is_throw else f"ACTION:{action}")
            if reply == "ACK":
                if is_throw:
                    self.last_throw_e_hp = float(before["obs"][47])
                    self.balls_thrown_now += 1
                    self.balls_left = max(0, self.balls_left - 1)
                break
            self.consecutive_rejects += 1
            reject_retries += 1
            print(f"Bridge rejected ACTION:{action} ({reply}) -- retry {reject_retries}")
            try:
                rej_phase, _ = self._status()
            except Exception:
                rej_phase = "?"
            log_reject(self.episode, self.step_count, action, reply, rej_phase,
                       self.action_masks(refresh=False), reject_retries)
            if self.consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                self.last_obs = before["obs"]
                return before["obs"], 0.0, False, True, {"rejected": reply}
            time.sleep(POLL_S * 4)
            phase, _ = self._wait_for_phase(READY_PHASES | END_PHASES, self.turn_timeout)
            if phase is None:
                return self._timeout_return(action, before["obs"])
            if phase in END_PHASES:
                st = self._get_state()
                return st["obs"], 0.0, True, False, {"player_lost": st["player_lost"], "outcome": st["outcome"]}
            forced_switch_turn = (phase == "FORCED_SWITCH")
            before = self._get_state()
            prev_e1_pp = [round(before["obs"][48 + i] * 40.0) for i in range(4)]
            prev_e1_moves = [round(before["obs"][52 + i] * 355.0) for i in range(4)]
            if reject_retries >= 2:   # same action keeps failing: swap in a legal one
                legal = np.flatnonzero(self.action_masks())
                if len(legal):
                    action = int(legal[0])
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
        balls_thrown = 0
        level_gain = 0.0
        new_species = 0
        if terminated:
            reward += self._outcome_bonus(st["outcome"])
            if st["outcome"] == OUTCOME_WON and self.start_p_team:
                reward += WIN_HP_BONUS * min(1.0, st["p_team"] / self.start_p_team)
            # Charge for balls used this battle (also counts manual throws).
            inv_end = self._read_inventory()
            start, end = self.battle_start.get("balls"), self._ball_total(inv_end)
            if start is not None and end is not None:
                balls_thrown = max(0, start - end)
                reward -= BALL_COST * balls_thrown
            # Party growth this battle, log-scaled so levels on weak members pay more.
            if inv_end and self.level_sum_start is not None:
                lv_end = [int(x) for x in inv_end["lv"].split(",") if x]
                if lv_end:
                    level_gain = self._log_level_sum(lv_end) - self.level_sum_start
                    reward += LEVEL_REWARD * level_gain
                    if level_gain:   # a level-up (or a swap/faint changing the party): show the inputs
                        print(f"[level] party {self.level_start} -> {lv_end}: "
                              f"gain {level_gain:+.4f} x {LEVEL_REWARD:.0f} = {LEVEL_REWARD * level_gain:+.2f} reward")
            if st["outcome"] == OUTCOME_CAUGHT:
                reward += self._catch_value()
                new_species = int(self.enemy_owned is False)
                self.box_cache = None   # the new Pokemon changes the PC
                log_catch(self.episode, SPECIES_ID_TO_NAME.get(before["e_species"], before["e_species"]),
                          before["e_level"], before["p_level"], None, balls_thrown, "manual")

        # Faint bonus/penalty (alive-count change; switching never changes it).
        reward += KO_BONUS * max(0, self.prev_e_alive - st["e_alive"])
        reward -= KO_BONUS * max(0, self.prev_p_alive - st["p_alive"])

        action_failed = (result == "FAILED")
        if action_failed and not is_attack_action and not is_throw:
            self.switch_blocked = True   # trapped by something we can't see
        if not forced_switch_turn:
            reward -= TURN_COST          # stalling is never free
            if not is_attack_action and not is_throw and not action_failed:
                # Voluntary switch; switch_count = switches already made this battle.
                # Free when it brings a lower-level member into a fight the lead is far
                # too strong for: that is exactly the training the bench needs.
                target_lv = self.party_levels[action - 11] if 12 <= action <= 16 else 0
                gap = before["p_level"] - before["e_level"]
                if not (gap >= SWITCH_FREE_GAP and 0 < target_lv < before["p_level"]):
                    reward -= SWITCH_COST * (1.0 + SWITCH_ESCALATION * self.switch_count)

        self.prev_p_team, self.prev_e_team = st["p_team"], st["e_team"]
        self.prev_p_alive, self.prev_e_alive = st["p_alive"], st["e_alive"]
        self.total_reward += reward
        self.session_reward += reward
        if is_attack_action:
            self.move_count += 1
        elif is_throw:
            pass   # counted through balls_thrown_now
        elif forced_switch_turn:
            self.forced_switch_count += 1
        elif action_failed:
            self.failed_switch_count += 1
        else:
            self.switch_count += 1
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
            outcome_txt = f" | {outcome_name(st['outcome'])}"
        else:
            outcome_txt = ""
        failed_txt = " | ACTION FAILED" if action_failed else ""
        if not action_failed and getattr(self, 'last_trapped', False):
            failed_txt += " | TRAPPED (switches masked)"

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
        self.step_count += 1
        info = {"player_lost": player_lost, "outcome": st["outcome"], "action_failed": action_failed}
        if terminated:
            info["battle_stats"] = {
                "voluntary_switches": self.switch_count,
                "forced_switches": self.forced_switch_count,
                "failed_switches": self.failed_switch_count,
                "moves": self.move_count,
                "turns": self.step_count,
                "return": self.total_reward,
                "lead_level": self.battle_start["lead_level"],
                "enemy_level": self.battle_start["enemy_level"],
                "level_gap": self.battle_start["lead_level"] - self.battle_start["enemy_level"],
                "trainer_battle": self.battle_start["trainer"],
                "balls_thrown": balls_thrown,
                "caught": int(st["outcome"] == OUTCOME_CAUGHT),
                "new_species": new_species,
                "level_gain": level_gain,
                "agent_throws": self.balls_thrown_now,
                # Scripted catching since the previous finished battle (see reset()).
                "scripted_balls": self.scripted_balls,
                "scripted_caught": self.scripted_caught,
                "swaps": self.swaps,
            }
            self.scripted_balls = self.scripted_caught = 0
            self.swaps = 0
            log_battle_end(self.episode, st["outcome"], self.step_count - 1, self.total_reward,
                           info["battle_stats"])
            log_battle_row(self.episode, st["outcome"], self.step_count, self.total_reward,
                           info["battle_stats"])
        # The PPO critic sees REWARD_SCALE * reward; everything logged/printed stays unscaled.
        return obs, reward * REWARD_SCALE, terminated, False, info

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
        print(f"\n=== Session summary ===\nTotal reward: {self.session_reward:+.1f}")
        self.sock.close()
