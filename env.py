import socket
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PokemonBattleEnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=8888, action_wait=2.0):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock_file = self.sock.makefile("r")
        self.action_wait = action_wait  # seconds to wait for a move to resolve

        self.action_space = spaces.Discrete(4)  # 4 moves
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        self.prev_player_hp = None
        self.prev_enemy_hp = None

    def _get_state(self):
        self.sock.sendall(b"GET_STATE\n")
        raw = self.sock_file.readline()
        parts = [int(x) for x in raw.strip().split(",")]
        player_hp, player_max, enemy_hp, enemy_max, battle_over = parts
        obs = np.array([
            player_hp / max(player_max, 1),
            enemy_hp / max(enemy_max, 1)
        ], dtype=np.float32)
        return obs, player_hp, enemy_hp, bool(battle_over)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            obs, p_hp, e_hp, _ = self._get_state()
            if e_hp > 0:
                break
            time.sleep(0.5)

        self.prev_player_hp = p_hp
        self.prev_enemy_hp = e_hp
        return obs, {}

    def step(self, action):
        # 1. Send the action (DQN decided this, not us)
        self.sock.sendall(f"ACTION:{action}\n".encode())
        ack = self.sock_file.readline()  # just consume the "ACK\n"

        # 2. Wait for the button-press sequence + battle animation to resolve
        time.sleep(self.action_wait)

        # 3. NOW poll for the resulting state
        obs, p_hp, e_hp, battle_over = self._get_state()

        # 4. Compute reward from the change
        reward = (self.prev_enemy_hp - e_hp) - (self.prev_player_hp - p_hp)
        if battle_over:
            reward += 100 if e_hp <= 0 else -100

        self.prev_player_hp = p_hp
        self.prev_enemy_hp = e_hp

        terminated = battle_over
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.sock.close()

# --- Quick sanity test ---
if __name__ == "__main__":
    env = PokemonBattleEnv()
    obs, info = env.reset()
    print("Initial obs:", obs)

    obs, reward, terminated, truncated, info = env.step(0)
    print("After action 0:", obs, reward, terminated)

    env.close()
