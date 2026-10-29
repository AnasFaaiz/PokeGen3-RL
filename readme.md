# PokeGen3-RL

A reinforcement-learning agent that plays Pokémon Emerald battles, trained with
[MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
(Stable-Baselines3 + sb3-contrib) against the live game running in
[mGBA](https://mgba.io/), driven through a Lua bridge over TCP.

The agent reads real game memory (party stats, HP, PP, cursors, battle
state) each turn, picks an action under an action mask, and the bridge
translates that action into the exact button presses the game needs —
built from the game's own state flags (`gMain.inBattle`, the active
controller function, the battle-script opcode) rather than timed key
sequences, so it never presses buttons the game isn't actually waiting on.

## How it fits together

```
┌─────────────┐   TCP :8888   ┌────────────────────┐   Lua/C API   ┌───────┐
│  env.py      │◄─────────────►│ battle_bridge.lua  │◄─────────────►│ mGBA  │
│ (Gym env)    │   TCP :8889   │ (game-state driven) │   emu:read*   │ + ROM │
└──────┬───────┘                └──────────┬─────────┘               └───────┘
       │                                   │
       ▼                                   ▼
 train_dqn.py                      tools/mem_finder.py
 (MaskablePPO)                     (automatic RAM address finder)
```

- **`env.py`** — a Gymnasium environment. Talks to the bridge over a plain
  TCP socket, builds the 417-value observation vector, computes reward, and
  exposes the action mask MaskablePPO needs.
- **`games/emerald/battle_bridge.lua`** — runs inside mGBA. Reads RAM
  directly, drives menu navigation by reading the actual cursor position
  after each press (not blind timing), and reports a `phase` (`CHOOSE_ACTION`,
  `WAITING`, `BATTLE_OVER`, `FORCED_SWITCH`, …) that `env.py` waits on.
- **`games/emerald/emerald_config.lua`** — every RAM address and struct
  offset used, pulled from the `pokeemerald` decompilation source/symbols.
- **`training/train_dqn.py`** — the MaskablePPO training loop, with a
  milestone checkpoint every 2048 steps and a crash-safety checkpoint
  (`backups/latest_safety.zip`) every 256 steps.
- **`training/expand_model.py`** — expands an older checkpoint's first
  layer when the observation size grows (used to go from 285 → 417 dims),
  without needing a live mGBA connection.
- **`tools/mem_finder.py`** — connects to the bridge's probe port and
  automatically finds RAM addresses: it looks for bytes that mirror known
  signals, bytes that discriminate between game states, and prompts the
  bridge doesn't yet handle. See [Memory address finder](#memory-address-finder).

## Setup

**Requirements**
- Python 3.10+
- [mGBA](https://mgba.io/) with Lua scripting support
- A legally-obtained Pokémon Emerald ROM (not included)
- `pip install -r requirements.txt` (stable-baselines3, sb3-contrib,
  gymnasium, numpy)

**1. Load the bridge in mGBA**

Open your ROM in mGBA, then Tools → Script Viewer → Load script →
`games/emerald/battle_bridge.lua`. The console should print:

```
Bridge listening on 8888 (env) and 8889 (address finder)
```

**2. Train**

```bash
python training/train_dqn.py
```

This loads `checkpoints/current/pokemon_ppo_masked_v7_expanded.zip` if it
exists (otherwise starts a fresh model), and trains for 500,000 steps.
TensorBoard logs go to `ppo_tensorboard/`.

**3. Evaluate / run a trained model**

```bash
python training/run_eval.py
```

## Observation space

417 values, each normalized to `[0, 1]`:

| Group | Contents |
|---|---|
| Battler state (×4: P1/E1/P2/E2) | level, HP ratio, stats, types, status, stat stages, ability, PP, moves |
| Own party (×6, active-first) | species, level, HP ratio, status |
| Enemy party (×6) | species, level, HP ratio, status |
| Battle context | weather category, battle-type flags, outcome/forced-switch flags |

## Reward

- **Damage dealt / taken**: scaled by the *fraction of max team HP* changed
  (not raw HP), so a knockout is worth roughly the same regardless of level.
- **Win / loss**: ±100, read directly from `gBattleOutcome`.
- **Switching**: no reward by itself. An escalating penalty applies from the
  *second* consecutive switch onward, so one tactical switch is free but
  stalling by repeated switching is discouraged.
- **Timeouts / bridge failures**: episode ends `truncated` with reward 0 —
  never penalized, since it's not something the model controlled.

## Memory address finder

`tools/mem_finder.py` runs alongside training or manual play and connects to
the bridge's probe port (8889, read-only — it never presses anything).

```bash
python tools/mem_finder.py watch --sym path/to/pokeemerald.sym
```

It records a RAM snapshot whenever the game enters a state it hasn't seen,
and every ~10 minutes writes `analysis/mem_sessions/<timestamp>/report.md`
covering:

1. **Signal mirrors** — addresses whose value always matches a signal the
   bridge already reports (confirms/duplicates known addresses).
2. **State discriminators** — addresses that are constant within one game
   state and differ between states (finds flags like screen callbacks or
   menu state).
3. **Unidentified waits** — prompts the game sat on that the bridge doesn't
   have a handler for yet.
4. **Config check** — re-tests every address in `emerald_config.lua`
   against data that doesn't come from that same address.

Pass `--sym pokeemerald.sym` to get readable names (`gBattleMons+0x28`)
instead of raw addresses, and `--pret path/to/pokeemerald` to name
battle-script opcodes. Results are ranked candidates, not proof — confirm
before changing the config.

```bash
# split recording and analysis
python tools/mem_finder.py record
python tools/mem_finder.py analyze analysis/mem_sessions/<session> --sym pokeemerald.sym
```

## Project layout

```
games/emerald/          battle_bridge.lua, emerald_config.lua, game data (species/moves/types)
training/                env.py, train_dqn.py, expand_model.py, logger.py, run_eval.py
tools/                   mem_finder.py
checkpoints/current/     the model currently being trained/loaded
backups/                 auto-generated milestone + crash-safety checkpoints (not committed)
analysis/mem_sessions/   mem_finder.py output (not committed)
```

## Status / caveats

- The bridge and `env.py` rewrite (game-state-driven battle detection,
  cursor-verified navigation, PID-based switching) has only been verified
  against a simulated battle engine built from the `pokeemerald` source —
  not yet confirmed stable on a long real-mGBA run.
- `checkpoints/current/pokemon_ppo_masked_v7.zip` was trained on an earlier,
  buggy observation/reward setup (swapped level/max_hp offsets, wrong
  weather address, switch-reward exploit) — expect a relearning period
  after loading the expanded checkpoint.

## License

See [LICENSE](LICENSE).
