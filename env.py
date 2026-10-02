import socket
import time
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from logger import init_csv, log_step, log_battle_end

# --- Load move type lookup once, at import time ---
with open("emerald_moves.json") as f:
    _raw_moves = json.load(f)
MOVE_TYPES = {int(k): v["type_id"] for k, v in _raw_moves.items()}

# --- Load type effectiveness chart once, at import time ---
with open("type_chart.json") as f:
    _raw_type_chart = json.load(f)
TYPE_CHART = {
    int(atk): {int(dfd): mult for dfd, mult in defenders.items()}
    for atk, defenders in _raw_type_chart.items()
}

def get_effectiveness(move_type, defender_type1, defender_type2):
    mult1 = TYPE_CHART.get(move_type, {}).get(defender_type1, 1.0)
    if defender_type2 != defender_type1:
        mult2 = TYPE_CHART.get(move_type, {}).get(defender_type2, 1.0)
    else:
        mult2 = 1.0
    return mult1 * mult2

# --- Status1 bitmasks ---
STATUS1_SLEEP_MASK = 0x07
STATUS1_PARALYSIS = 0x40
STATUS1_FREEZE = 0x20


class PokemonBattleEnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=8888, action_wait=2.0):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock_file = self.sock.makefile("r")
        self.action_wait = action_wait

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(34,), dtype=np.float32)

        self.prev_player_hp = None
        self.prev_enemy_hp = None

        init_csv()
        self.episode = 0
        self.step_count = 0

    def _get_state(self):
        self.sock.sendall(b"GET_STATE\n")
        raw = self.sock_file.readline()
        parts = [int(x) for x in raw.strip().split(",")]

        (player_hp, player_max, enemy_hp, enemy_max, battle_over, player_lost,
         pp1, pp2, pp3, pp4, move1, move2, move3, move4,
         p_type1, p_type2, e_type1, e_type2, p_level, e_level,
         p_status1, e_status1, p_status2, e_status2,
         p_atk, p_def, p_spd, e_atk, e_def, e_spd) = parts

        move1_type = MOVE_TYPES.get(move1, 17)
        move2_type = MOVE_TYPES.get(move2, 17)
        move3_type = MOVE_TYPES.get(move3, 17)
        move4_type = MOVE_TYPES.get(move4, 17)

        move1_eff = get_effectiveness(move1_type, e_type1, e_type2)
        move2_eff = get_effectiveness(move2_type, e_type1, e_type2)
        move3_eff = get_effectiveness(move3_type, e_type1, e_type2)
        move4_eff = get_effectiveness(move4_type, e_type1, e_type2)

        obs = np.array([
            player_hp / max(player_max, 1),
            enemy_hp / max(enemy_max, 1),
            pp1 / 40.0, pp2 / 40.0, pp3 / 40.0, pp4 / 40.0,
            move1 / 355.0, move2 / 355.0, move3 / 355.0, move4 / 355.0,
            move1_type / 17.0, move2_type / 17.0, move3_type / 17.0, move4_type / 17.0,
            move1_eff / 4.0, move2_eff / 4.0, move3_eff / 4.0, move4_eff / 4.0,
            p_type1 / 18.0, p_type2 / 18.0, e_type1 / 18.0, e_type2 / 18.0,
            p_level / 100.0, e_level / 100.0,
            p_status1 / 255.0, e_status1 / 255.0,
            p_status2 / 4294967295.0, e_status2 / 4294967295.0,
            p_atk / 12.0, p_def / 12.0, p_spd / 12.0,
            e_atk / 12.0, e_def / 12.0, e_spd / 12.0
        ], dtype=np.float32)

        return obs, player_hp, enemy_hp, bool(battle_over), bool(player_lost), [pp1, pp2, pp3, pp4], p_status1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_count = 0
        while True:
            obs, p_hp, e_hp, _, _, _, _ = self._get_state()
            if e_hp > 0 and p_hp > 0:
                break
            time.sleep(0.5)

        self.prev_player_hp = p_hp
        self.prev_enemy_hp = e_hp
        return obs, {}

    def step(self, action):
        obs, p_hp, e_hp, battle_over, player_lost, pp_list, p_status1 = self._get_state()

        if e_hp <= 0 or p_hp <= 0:
            print("[WARNING] step() called but doesn't look like an active battle — skipping")
            return obs, 0, True, False, {"player_lost": player_lost}

        print(f"DEBUG | Action Chosen: {action} | Current PP List: {pp_list} | Total PP: {sum(pp_list)}")

        is_attack_action = action < 4
        out_of_all_pp = sum(pp_list) <= 0

        if is_attack_action and not out_of_all_pp and pp_list[action] <= 0:
            reward = -5
            log_step(self.episode, self.step_count, action, obs, reward, False, False)
            self.step_count += 1
            return obs, reward, False, False, {"player_lost": False}

        turn_resolved = False
        initial_pp_sum = sum(pp_list)
        initial_p_hp = p_hp
        initial_e_hp = e_hp

        max_outer_retries = 5
        outer_attempts = 0

        current_p_hp, current_e_hp = p_hp, e_hp
        current_battle_over, current_player_lost, current_pp_list = battle_over, player_lost, pp_list
        current_p_status1 = p_status1

        while not turn_resolved and outer_attempts < max_outer_retries:
            outer_attempts += 1
            self.sock.sendall(f"ACTION:{action}\n".encode())
            ack = self.sock_file.readline()

            timeout_counter = 0
            max_timeout = 20

            while timeout_counter < max_timeout:
                time.sleep(0.1)
                timeout_counter += 1
                obs, current_p_hp, current_e_hp, current_battle_over, current_player_lost, current_pp_list, current_p_status1 = self._get_state()

                if current_p_hp <= 0 and current_e_hp <= 0:
                    print(f"[FLEE-SUSPECT] After action {action}, both HP readings are 0 (attempt {outer_attempts})")

                if sum(current_pp_list) < initial_pp_sum or current_p_hp != initial_p_hp or current_e_hp != initial_e_hp or current_battle_over:
                    turn_resolved = True
                    break

        if not turn_resolved:
            is_paralyzed = (current_p_status1 & STATUS1_PARALYSIS) != 0
            is_asleep = (current_p_status1 & STATUS1_SLEEP_MASK) != 0
            is_frozen = (current_p_status1 & STATUS1_FREEZE) != 0

            if is_paralyzed or is_asleep or is_frozen:
                print(f"[STATUS-SKIP] Turn didn't resolve, but status1={current_p_status1} explains it — legitimate skipped turn")
                reward = 0
                log_step(self.episode, self.step_count, action, obs, reward, False, False)
                self.step_count += 1
                return obs, reward, False, False, {"player_lost": False}

            print(f"[WARNING] Turn never resolved after {outer_attempts} attempts — treating as invalid, ending episode")
            reward = -20
            log_step(self.episode, self.step_count, action, obs, reward, True, False)
            log_battle_end(self.episode, False)
            self.step_count += 1
            return obs, reward, True, False, {"player_lost": False}

        p_hp = current_p_hp
        e_hp = current_e_hp
        battle_over = current_battle_over
        player_lost = current_player_lost
        pp_list = current_pp_list

        reward = (self.prev_enemy_hp - e_hp) - (self.prev_player_hp - p_hp)
        if battle_over:
            if player_lost:
                reward -= 100
            else:
                reward += 100

        self.prev_player_hp = p_hp
        self.prev_enemy_hp = e_hp

        log_step(self.episode, self.step_count, action, obs, reward, battle_over, player_lost)

        if battle_over:
            won = not player_lost
            log_battle_end(self.episode, won)

        self.step_count += 1

        return obs, reward, battle_over, False, {"player_lost": player_lost}

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    env = PokemonBattleEnv()
    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Obs length:", len(obs))

    obs, reward, terminated, truncated, info = env.step(0)
    print("After action 0:", obs, reward, terminated)

    env.close()
