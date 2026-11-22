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
  TCP socket, builds the observation vector (499 values by default), computes reward, and
  exposes the action mask MaskablePPO needs.
- **`games/emerald/battle_bridge.lua`** — runs inside mGBA. Reads RAM
  directly, drives menu navigation by reading the actual cursor position
  after each press (not blind timing), and reports a `phase` (`CHOOSE_ACTION`,
  `WAITING`, `BATTLE_OVER`, `FORCED_SWITCH`, …) that `env.py` waits on.
- **`games/emerald/emerald_config.lua`** — every RAM address and struct
  offset used, pulled from the `pokeemerald` decompilation source/symbols.
- **`training/train_dqn.py`** — the MaskablePPO training loop (despite the
  name, no DQN any more). It loads `pokemon_ppo_masked_v8` and saves
  `pokemon_ppo_masked_v9`, with a crash-safety checkpoint
  (`backups/latest_safety.zip`) and optional parallel emulators.
- **`training/expand_model*.py`** — expand an older checkpoint's first layer
  (and action head) when the observation or action space grows
  (285 → 417 → 431 → 499 dims), without a live mGBA connection. Old weights
  keep their meaning; new inputs start at zero.
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

Build the v9 checkpoint from v8 once (no emulator needed), then train:

```bash
python training/expand_model_v9.py
python training/train_dqn.py
```

This trains for 500,000 steps by default (`POKE_TOTAL_STEPS`). TensorBoard
logs go to `ppo_tensorboard/`. Flags such as `POKE_N_ENVS` (parallel
emulators), `POKE_MOVE_TEACH`, `POKE_FRAME_STACK` and `POKE_LEGACY_OBS` are
listed in [PLAN.md](PLAN.md). Older models are on the
[Releases page](https://github.com/AnasFaaiz/PokeGen3-RL/releases) (DQN v1–v6,
PPO v7, v7-expanded, v8).

**3. Evaluate / run a trained model**

```bash
python analysis/run_eval.py --model checkpoints/current/pokemon_ppo_masked_v9 --battles 100
```

## Observation space

499 values by default (431 with `POKE_LEGACY_OBS=1`, which is what v8 was trained on), each normalized to `[0, 1]`:

| Group | Contents |
|---|---|
| Battler state (×4: P1/E1/P2/E2) | level, HP ratio, stats, types, status, stat stages, ability, PP, moves |
| Own party (×6, active-first) | species, level, HP ratio, status |
| Enemy party (×6) | species, level, HP ratio, status |
| Battle context | weather category, battle-type flags, outcome/forced-switch flags |
| Training context (v8, +14) | party levels, level gap, balls left, wild flag, species already owned, PC stock |
| Combat features (v9, +60) | expected damage, KO flags, speed, stats, bench, move tags |
| Opponent trainer (v9, +8) | trainer team features from the decomp tables |

## Reward

- **Damage dealt / taken**: scaled by the *fraction of max team HP* changed
  (not raw HP), so a knockout is worth roughly the same regardless of level.
- **Win / loss**: ±100 (win bonus adjustable via `POKE_WIN_REWARD`), read directly from `gBattleOutcome`.
- **KO bonus**: +5 per enemy faint, -5 per faint of your own.
- **Level growth**: a log-scale term on party level gain, so early levels pay more than late ones.
- **Costs**: -1 per decision turn and -3 per *voluntary* switch, escalating
  by 50% for each further switch (max 4 per battle). Forced switches (after a
  faint) and switches the game refused are free. Running from a wild battle
  costs `POKE_FLEE_PENALTY` (40).
- **Catching** (outcome 7): the model's own action 17 throws a ball (wild
  battles only, max 4 per battle). Catches pay a bonus, more for a new Pokedex
  species or a low-HP catch, minus a per-ball cost.
- **Scaling**: all rewards are multiplied by 0.01 before PPO sees them.
- **Rejected actions**: never stored as training steps. The env retries the
  action, then falls back to a legal one, and logs each reject to
  `logs/rejects.csv`.
- **Timeouts / bridge failures**: episode ends `truncated` with reward 0 —
  never penalized, since it's not something the model controlled.

## Catching

Catching is now a model action (action 17, `THROW_ACTION`). The scripted rule
below (`training/catch_policy.py`) is dormant (`SCRIPTED_CATCH = False`) and
kept for reference; it keeps a pool of low-level Pokemon on hand for when the
party has outleveled the area it hunts in.

- **Trigger:** every party member is at least 12 levels above the highest
  wild level recently seen on this map (last 20 encounters, at least 10
  needed), and the PC holds fewer than 6 Pokemon at or just above that level.
- **Action:** catch the wild Pokemon, up to 4 balls each, skipping species
  already in the PC. Wild battles only.
- **When it runs:** in `reset()`, before the agent's first move, so no action
  the policy chose is ever replaced. If a ball catches it, `reset()` moves on
  to the next encounter. The agent is not charged for HP lost or balls spent
  during those throws.

Set `SCRIPTED_CATCH = False` in `env.py` to turn it off (`run_eval.py` already
does). Logs: `logs/catches.csv` (every catch), `logs/throws.csv` (every throw and its
result), `logs/battles.csv` (one row per battle: outcome, wild or trainer,
level gap, switch counts, balls) and `logs/level_gap.csv` (party-vs-PC levels
every 25 battles). `battle_data.csv` is the per-step log; if its header is
older than the current columns, the old file is renamed to
`battle_data_oldschema_<time>.csv` when training starts. Every row carries a
run id (`logs/runs.csv`); `python training/log_report.py` summarises logs per
run, level gap and species.

## PC-box swap

`training/swap_policy.py` swaps overleveled party members for PC Pokemon near
the wild area's level, so fights stay informative. It writes party and box
memory directly in the overworld (`BOXMON` / `PARTYMON` / `SWAP` bridge
commands), not by driving the PC menu. The whole party is checked, a start-up
fill brings it into the area's band, picks are weighted so species rotate, box
eggs are skipped and computed stats are sanity-checked first. Wild areas only
(needs 10 wild encounters on a map first); turn off with `SCRIPTED_SWAP = False`
for gym or story battles. Verified live. Swaps are logged in `logs/swaps.csv`.

## Move teaching (optional)

With `POKE_MOVE_TEACH=1`, four extra actions (18–21) let the model choose a
level-up move to teach a party member, scored by `training/move_policy.py`
and written between battles by `SETMOVES`. Not yet checked on the real game;
see [PLAN.md](PLAN.md).

Read-only and manual bridge commands (`python training/bridge_cli.py <CMD>`,
training stopped):

| Command | What it does |
|---|---|
| `INVENTORY` | money, ball counts, party levels, last outcome, trainer-battle flag |
| `BOX` | every occupied PC slot with species and level |
| `DEX` | Pokedex owned flags |
| `BAG` | bag pocket/cursor and current callback |
| `THROW[:itemId]` | throw a ball in a wild single battle (default Poke Ball) |
| `PARTYMON` / `BOXMON` / `SWAP` / `SPECIES` | read party/box Pokemon, swap a party slot with a box slot |
| `MONMOVES` / `SETMOVES` | read and write a party Pokemon's moves (move teaching) |
| `TRAINER` / `TRAINERSCAN` / `TRAINERFLAGS` | opponent trainer lookup and defeated flags (address not yet found live) |

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
games/battle_bridge.lua  the mGBA bridge (battle tasks, throw task, socket server)
games/bridge_ports.lua   picks a free port pair per emulator window
games/emerald/           emerald_config.lua, inventory.lua (party/box/move readers and writers),
                         overworld_nav.lua, game_data/ (decomp-generated tables), data_gen/ (generators)
training/                env.py, train_dqn.py, swap_policy.py, catch_policy.py, move_policy.py,
                         damage_calc.py, trainer_info.py, expand_model*.py, logger.py, log_report.py,
                         prepare_instances.py, check_ports.py, bridge_cli.py, test_*.py (offline tests)
analysis/                run_eval.py, reinit_value_head.py, distill_policy.py (mem_sessions/ not committed)
checkpoints/current/     v7-expanded, v8 and v9 models
backups/                 auto-generated milestone + crash-safety checkpoints (not committed)
PLAN.md                  current work, env flags, live-test checklist
```

## Status / caveats

- The battle bridge and `env.py` are driven by game state (battle detection,
  cursor-verified navigation, PID-based switching). Catching and the PC-box swap
  have run live. Move teaching, trainer-id lookup and the v9 observation are
  verified only offline so far; the live checklist is in [PLAN.md](PLAN.md).
- Training speed on one emulator is about 1,000 steps/hour; parallel emulators
  (`POKE_N_ENVS`) are the main lever.
- The v7 checkpoints were trained on an earlier, buggy observation/reward setup
  (swapped level/max_hp offsets, wrong weather address, switch-reward exploit),
  so v8 and later are the ones to use.

## License

See [LICENSE](LICENSE).
