"""Env-level checks for move teaching (POKE_MOVE_TEACH=1), with a stubbed bridge: POKE_MOVE_TEACH=1 python training/test_teach_env.py
Without the flag it only checks that nothing about the default env changed."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env as e
import logger
import move_policy as mp
import mon_struct

if not e.MOVE_TEACH:
    expect = 431 if e.LEGACY_OBS else 499 + (61 if e.FRAME_STACK else 0)
    assert e.N_ACTIONS == 18 and e.TEACH_FEATURES == 0 and e.OBS_DIM == expect, (e.OBS_DIM, expect)
    print("ok (flag off: default env unchanged)")
    sys.exit(0)

E = e.PokemonBattleEnv
NAMES = {v["name"]: int(k) for k, v in mp._moves().items()}
SP = {v: int(k) for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games", "emerald",
                                                          "game_data", "species_id_to_name.json"))).items()}
assert e.N_ACTIONS == 22 and e.TEACH_FEATURES == 44 and e.TEACH_BASE == 18


class FakeGame:
    """Answers MONMOVES / SETMOVES like the Lua does, for a small party kept in memory."""
    def __init__(self, mons):
        self.mons = mons                    # list of dicts: pid, species, level, moves, pp, bonuses, ok
        self.sent = []

    def request(self, line):
        self.sent.append(line)
        if line.startswith("MONMOVES:"):
            slot = int(line.split(":")[1])
            if slot >= len(self.mons):
                return "MONMOVES:ERR:bad_slot"
            m = self.mons[slot]
            return "MONMOVES:%d:%d:%d:%s:%s:%d:%d" % (m["pid"], m["species"], m["level"], ",".join(map(str, m["moves"])),
                                                       ",".join(map(str, m["pp"])), m["bonuses"], 1 if m["ok"] else 0)
        if line.startswith("SETMOVES:"):
            t = line.split(":")
            slot, pid = int(t[1]), int(t[2])
            if self.mons[slot]["pid"] != pid:
                return "SETMOVES:ERR:pid_mismatch"
            self.mons[slot]["moves"] = [int(x) for x in t[3:7]]
            self.mons[slot]["pp"] = [int(x) for x in t[7:11]]
            return "SETMOVES:OK"
        raise AssertionError(line)


def make_env(game):
    env = object.__new__(E)
    env._request = game.request
    env._wait_for_overworld = lambda: True
    env._get_state = lambda: {"obs": np.zeros(e.OBS_DIM, dtype=np.float32)}
    env.episode, env.total_reward, env.session_reward = 1, 0.0, 0.0
    env.last_obs = np.zeros(e.OBS_DIM, dtype=np.float32)
    env._init_teach_state()
    return env


mudkip = {"pid": 1001, "species": SP["MUDKIP"], "level": 30, "moves": [NAMES["Tackle"], NAMES["Growl"], NAMES["Mud Slap"], NAMES["Water Gun"]],
          "pp": [35, 40, 10, 25], "bonuses": 0b00010100, "ok": True}
treecko = {"pid": 2002, "species": SP["TREECKO"], "level": 25, "moves": [NAMES["Pound"], NAMES["Leer"], NAMES["Absorb"], NAMES["Quick Attack"]],
           "pp": [35, 30, 20, 30], "bonuses": 0, "ok": True}
bad = {"pid": 3003, "species": SP["TORCHIC"], "level": 16, "moves": [NAMES["Scratch"], NAMES["Growl"], 0, 0], "pp": [35, 40, 0, 0],
       "bonuses": 0, "ok": False}

with tempfile.TemporaryDirectory() as d:
    logger.MOVES_FILE = os.path.join(d, "moves.csv")
    logger._checked.clear()

    # Closed window: features are all zero and no teach action is legal.
    game = FakeGame([dict(mudkip), dict(treecko), dict(bad)])
    env = make_env(game)
    assert env._teach_features() == [0.0] * 44 and not env.teach_open

    # Opening: the mon with the biggest gain is the focus; a mon with a bad checksum is never chosen.
    assert env._open_teach_window() and env.teach_open
    assert env.teach_focus["pid"] in (1001, 2002) and 1 <= len(env.teach_cands) <= 4
    assert env.teach_cands == sorted(env.teach_cands, key=lambda c: (-c["delta"], c["move"]))
    f = env._teach_features()
    assert len(f) == 44 and all(0.0 <= v <= 1.0 for v in f) and f[0] == 1.0 and f[43] == 1.0, f
    for k in range(4):
        used = k < len(env.teach_cands)
        assert (f[7 + 9 * k] > 0) == used and (f[7 + 9 * k:16 + 9 * k] == [0.0] * 9) == (not used)

    # Mask: teach actions are legal exactly for the offered candidates, and only while the window is open and unused.
    mask_env = env
    mask_env.last_obs = np.zeros(e.OBS_DIM, dtype=np.float32)
    mask_env.last_obs[1:5] = 1.0
    mask_env.last_obs[5:9] = [NAMES["Tackle"] / 355.0] * 4
    mask_env.last_obs[13:17] = 0.25
    mask_env.last_obs[47] = 1.0
    mask_env.last_party_raw = [0] * 96
    mask_env.switch_count, mask_env.is_wild, mask_env.balls_left, mask_env.balls_thrown_now = 0, False, 0, 0
    mask_env.last_trapped = mask_env.switch_blocked = False
    mask_env.last_awaiting_forced_switch = False
    m = mask_env.action_masks(refresh=False)
    assert len(m) == 22 and list(m[18:]) == [k < len(env.teach_cands) for k in range(4)]
    assert m[0], "normal moves stay legal while the window is open"

    # Choosing: queues a request keyed by pid, pays shaping from the gain, costs no turn, never ends the battle.
    first = env.teach_cands[0]
    focus_pid = env.teach_focus["pid"]
    obs, rew, term, trunc, info = env._teach_step(0)
    assert not term and not trunc and info["action_failed"] is False and info["outcome"] == 0
    assert rew == 0.0 and env.total_reward == 0.0 and env._pending_shaping == 0.0      # nothing is paid at the choice
    assert env.teach_used == 1 and len(env.teach_queue) == 1
    req = env.teach_queue[0]
    assert req["pid"] == focus_pid and req["move"] == first["move"] and req["slot"] == first["slot"]
    assert env.teach_known[first["slot"]] == first["move"]                        # the plan continues from the new set
    assert all(c["move"] != first["move"] for c in env.teach_cands)
    # a second teach is allowed; after MAX the window offers nothing and an illegal pick is a small negative, not a crash
    if env.teach_cands:
        env._teach_step(0)
    env.teach_used = e.MAX_TEACH_PER_BATTLE
    env.teach_cands = []
    m = env.action_masks(refresh=False)
    assert not m[18:].any()
    obs, rew, term, trunc, info = env._teach_step(0)
    assert rew == -1.0 * e.REWARD_SCALE and info["action_failed"] is True and not term

    # Applying: the party was reordered (the mon is now in slot 2): the request is found by pid, PP of the new move is
    # its base PP plus the mon's PP Ups for that slot, the log row says OK, and the moves really changed.
    game2 = FakeGame([dict(treecko), dict(bad), dict(mudkip)])
    env2 = make_env(game2)
    env2.teach_queue = [{"pid": 1001, "slot": 1, "species": mudkip["species"], "level": 30, "known_before": list(mudkip["moves"]),
                         "move": NAMES["Take Down"], "replaced": NAMES["Growl"], "delta": 22.9}]
    env2._apply_teach_queue()
    cmd = [s for s in game2.sent if s.startswith("SETMOVES")][0]
    assert cmd.startswith("SETMOVES:2:1001:"), cmd                                 # slot looked up by pid, not remembered
    got = game2.mons[2]
    assert got["moves"][1] == NAMES["Take Down"] and got["moves"][0] == mudkip["moves"][0]
    assert got["pp"][1] == mon_struct.max_pp(mp.base_pp(NAMES["Take Down"]), mon_struct.clear_bonus(mudkip["bonuses"], 1), 1) and got["pp"][0] == 35
    assert env2.teach_queue == [] and env2.teach_applied == 1
    want = e.TEACH_SHAPING * float(np.clip(22.9 / 50.0, -1, 1))
    assert abs(env2._pending_shaping - want) < 1e-9 and env2.teach_fail_streak == 0      # paid only after the verified edit
    sent_bonus = int(cmd.split(":")[-1])
    assert sent_bonus == mon_struct.clear_bonus(mudkip["bonuses"], 1) == 0b00010000   # slot 1's PP-Up bits cleared, slot 2's kept
    assert game2.mons[2]["pp"][1] == mp.base_pp(NAMES["Take Down"])                      # no PP Ups left in that slot
    assert abs(env2._take_shaping() - want) < 1e-9 and env2._take_shaping() == 0.0 and abs(env2.teach_paid_battle - want) < 1e-9
    rows = list(__import__("csv").reader(open(logger.MOVES_FILE)))
    assert rows[1][rows[0].index("reply")] == "SETMOVES:OK" and rows[1][rows[0].index("in_move")] == "Take Down"

    # Stale or unusable requests: neutral skips cost nothing, a bad checksum counts as a failed edit path.
    game3 = FakeGame([dict(mudkip, moves=[NAMES["Tackle"], NAMES["Water Gun"], 0, 0]), dict(bad)])
    env3 = make_env(game3)
    base = {"slot": 1, "species": mudkip["species"], "level": 30, "move": NAMES["Take Down"], "replaced": NAMES["Growl"], "delta": 9.0}
    env3.teach_queue = [dict(base, pid=9999, known_before=list(mudkip["moves"])),         # mon gone: neutral
                        dict(base, pid=3003, known_before=list(bad["moves"]))]            # bad checksum: failure
    env3._apply_teach_queue()
    assert not any(s_.startswith("SETMOVES") for s_ in game3.sent) and env3.teach_applied == 0
    rows3 = list(__import__("csv").reader(open(logger.MOVES_FILE)))
    replies = [r[rows3[0].index("reply")] for r in rows3[2:]]
    assert replies == ["SKIP:mon_gone", "SKIP:bad_checksum"], replies
    assert env3.teach_fail_streak == 1 and abs(env3._pending_shaping + e.TEACH_FAIL_PENALTY) < 1e-9

    # Moves changed since the plan (a level-up learned a move in the battle): re-plan on the current set and apply the
    # move into the slot that is best NOW, or skip when it is no longer worth it.
    game3b = FakeGame([dict(mudkip, moves=[NAMES["Tackle"], NAMES["Water Gun"], NAMES["Mud Slap"], NAMES["Bide"]])])
    env3b = make_env(game3b)
    env3b.teach_queue = [dict(base, pid=1001, known_before=list(mudkip["moves"]), move=NAMES["Take Down"], delta=20.0)]
    env3b._apply_teach_queue()
    assert any(s_.startswith("SETMOVES:0:1001") for s_ in game3b.sent) and NAMES["Take Down"] in game3b.mons[0]["moves"]
    assert env3b.teach_applied == 1 and env3b._pending_shaping > 0
    game3c = FakeGame([dict(mudkip, moves=[NAMES["Water Gun"], NAMES["Take Down"], NAMES["Mud Slap"], NAMES["Tackle"]])])
    env3c = make_env(game3c)
    env3c.teach_queue = [dict(base, pid=1001, known_before=list(mudkip["moves"]), move=NAMES["Take Down"])]
    env3c._apply_teach_queue()                                                           # already known after the change
    assert not any(s_.startswith("SETMOVES") for s_ in game3c.sent) and env3c.teach_fail_streak == 0
    game3d = FakeGame([dict(mudkip, moves=[NAMES["Water Gun"], NAMES["Surf"], NAMES["Take Down"], NAMES["Hydro Pump"]])])
    env3d = make_env(game3d)
    env3d.teach_queue = [dict(base, pid=1001, known_before=list(mudkip["moves"]), move=NAMES["Growl"], delta=9.0)]
    env3d._apply_teach_queue()                                                           # a junk move into a strong set
    assert not any(s_.startswith("SETMOVES") for s_ in game3d.sent) and env3d.teach_fail_streak == 0

    # Circuit breaker: three failed edits in a row switch teaching off for the run (no window, no teach actions).
    class RefuseGame(FakeGame):
        def request(self, line):
            if line.startswith("SETMOVES:"):
                self.sent.append(line)
                return "SETMOVES:ERR:verify_failed"
            return super().request(line)
    gameb = RefuseGame([dict(mudkip)])
    envb = make_env(gameb)
    envb._wait_for_overworld = lambda: True
    for k in range(3):
        envb.teach_queue = [dict(base, pid=1001, known_before=list(mudkip["moves"]), move=NAMES["Take Down"], delta=20.0)]
        envb._apply_teach_queue()
        assert envb.teach_disabled == (k == 2), k
    assert envb._pending_shaping == -3 * e.TEACH_FAIL_PENALTY and envb.teach_applied == 0
    assert envb._open_teach_window() is False and not envb.teach_open
    # one success in between resets the streak
    gamer = FakeGame([dict(mudkip)])
    envr = make_env(gamer)
    envr.teach_fail_streak = 2
    envr.teach_queue = [dict(base, pid=1001, known_before=list(mudkip["moves"]), move=NAMES["Take Down"], delta=20.0)]
    envr._apply_teach_queue()
    assert envr.teach_fail_streak == 0 and not envr.teach_disabled

    # A full queue hides the teach actions instead of dropping requests silently.
    envq = make_env(FakeGame([dict(mudkip)]))
    assert envq._open_teach_window()
    envq.last_obs[1:5] = 1.0
    envq.last_obs[47] = 1.0
    envq.last_party_raw = [0] * 96
    envq.switch_count, envq.is_wild, envq.balls_left, envq.balls_thrown_now = 0, False, 0, 0
    envq.last_trapped = envq.switch_blocked = envq.last_awaiting_forced_switch = False
    assert envq.action_masks(refresh=False)[18:].any()
    envq.teach_queue = [dict(base, pid=1001, known_before=[0, 0, 0, 0])] * e.TEACH_QUEUE_MAX
    assert not envq.action_masks(refresh=False)[18:].any()
    obs_, rew_, *_rest, info_ = envq._teach_step(0)
    assert rew_ == -1.0 * e.REWARD_SCALE and info_["action_failed"] and len(envq.teach_queue) == e.TEACH_QUEUE_MAX

    # Not in the overworld yet: the queue is kept for next time.
    env4 = make_env(FakeGame([dict(mudkip)]))
    env4._wait_for_overworld = lambda: False
    env4.teach_queue = [dict(base, pid=1001, known_before=list(mudkip["moves"]))]
    env4._apply_teach_queue()
    assert len(env4.teach_queue) == 1

    # No window when nothing is worth teaching (a roster of one mon with a bad checksum), and no crash on bridge errors.
    env5 = make_env(FakeGame([dict(bad)]))
    assert env5._open_teach_window() is False and not env5.teach_open
    env6 = make_env(FakeGame([]))
    env6._request = lambda line: (_ for _ in ()).throw(OSError("closed"))
    assert env6._open_teach_window() is False
print("ok")
