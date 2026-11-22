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
                          ┌──────────────────────────────────────────────┐
                          │ train_dqn.py  (main process)                 │
                          │  MaskablePPO: ONE policy + value network     │
                          │  rollout buffer (n_steps x N envs)           │
                          │  callbacks: win-rate, milestone + safety     │
                          └───────┬──────────────────────────────▲───────┘
                  actions (batch) │   obs, reward, done, mask    │  weights saved to
                                  ▼   (SubprocVecEnv pipes)      │  checkpoints/ + backups/
        ┌─────────────┬───────────┴─┬─────────────┬──────────────┴┐
        │ worker 0    │ worker 1    │ worker 2    │ worker 3      │   env.py (Gymnasium env),
        │ env.py      │ env.py      │ env.py      │ env.py        │   no model inside: it only
        │ port 8888   │ port 8890   │ port 8892   │ port 8894     │   steps its own game
        └──────┬──────┴──────┬──────┴──────┬──────┴──────┬────────┘
          TCP  │        TCP  │        TCP  │        TCP  │   line protocol: GET_STATE, ACTION:n,
               ▼             ▼             ▼             ▼   THROW, NAV:*, SWAP, ...  (one reply per line)
        ┌────────────┐┌────────────┐┌────────────┐┌────────────┐
        │ mGBA #0    ││ mGBA #1    ││ mGBA #2    ││ mGBA #3    │  each: own ROM + .sav copy,
        │ battle_    ││ battle_    ││ battle_    ││ battle_    │  own battle_bridge.lua
        │ bridge.lua ││ bridge.lua ││ bridge.lua ││ bridge.lua │  (also probe port main+1 for
        └────────────┘└────────────┘└────────────┘└────────────┘   tools/mem_finder.py)
```

`POKE_N_ENVS=1` is the same picture with a single worker and no `SubprocVecEnv`.

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

## How the Lua bridge and the env work

**The bridge (`games/battle_bridge.lua`)** runs inside mGBA and is registered on the emulator's
`frame` callback, so it runs once per emulated frame. Each frame it:

1. Reads game RAM (addresses from `emerald_config.lua`): the main callback (`callback2`), `gMain.inBattle`,
   `gBattleOutcome`, the controller function of each battler and the battle-script opcode.
2. Works out a **phase** from those: `CHOOSE_ACTION`, `FORCED_SWITCH`, `WAITING`, `BUSY`, `BATTLE_OVER`,
   `NOT_IN_BATTLE`.
3. Advances the running **task** (one move, switch or ball throw) by at most one button press. Menus are
   navigated by pressing, then reading the real cursor position back and pressing again until it is right.
4. When no task is running, `battleAutomation` clears the prompts the agent never decides (evolution,
   "learn a move?", level-up boxes, post-battle text), with a watchdog that presses B if the game is stuck.
5. When out of battle, `overworld_nav.lua` walks the player to grass or to a Pokémon Center.
6. Services the TCP sockets: accepts the env's connection and answers one reply per command line.

The bridge is a plain request/response server. `GET_STATE` returns the whole battle state,
`STATUS` returns `phase:lastResult`, `ACTION:n` / `THROW` start a task and reply `ACK` or `REJECT:<why>`,
and `NAV:*`, `INVENTORY`, `BOX`, `SWAP`, `SETMOVES` and the rest are the overworld commands.

**The env (`training/env.py`)** is a standard Gymnasium environment where one episode is one battle:

- `reset()` waits for the previous battle to finish. In the overworld it applies queued move edits, runs the
  PC-box swap, then calls `NAV:HUNT`/`NAV:HEAL` and polls `STATUS` until the game is asking for an action.
- `step(action)` waits for a decision phase, snapshots the state, sends `ACTION:n`, polls until the next
  decision point or the end of the battle, and then builds the new observation, the reward and the mask.
  If the bridge rejects an action, the env retries it and falls back to a legal action. A rejected action is
  never stored as a training step.
- `action_masks()` marks illegal moves, switches and throws, and MaskablePPO only samples from the legal ones.
- A stuck emulator or a 60-second turn timeout ends the episode as `truncated` with reward 0.

## Four environments, one model

`POKE_N_ENVS=4` starts four **separate mGBA windows**, and the training process drives all of them:

1. `training/prepare_instances.py` gives each window its own copy of the ROM, `.sav` and start state, because
   two windows writing one save file would corrupt it. Each window loads `battle_bridge.lua`. The first free
   port pair is taken in order (`bridge_ports.lua`), so the windows get 8888/8889, 8890/8891, 8892/8893 and
   8894/8895. `python training/check_ports.py 4` checks that all four answer.
2. `train_dqn.py` wraps each env in a `SubprocVecEnv`, so each worker process owns one `PokemonBattleEnv` and
   one TCP connection to its own emulator. Workers hold no model.
3. There is a single `MaskablePPO` in the main process. Each step it takes the 4 observations and 4 masks as
   one batch, runs one forward pass and sends one action back to each worker. The four games then step in
   parallel, and each takes its own wall-clock time per turn.
4. Every env's transitions go into the **same rollout buffer**. `POKE_N_STEPS` is the total samples per update
   (default 512), so with 4 envs each contributes 128 steps. When the buffer is full, PPO does one update on all of it and
   the new weights are used by all four games on the next step. So the 4 envs are not 4 models; they are 4
   sources of experience for one policy, which is why the speed-up is close to linear in the emulator count.
5. Logging stays separable. Each worker gets its own log id (`<run>_e0` … `_e3`) and episode numbers start at
   `i * 10,000,000`, so rows from different emulators never collide.

### Commands for the 4-env run

```bash
# 1. One ROM + .sav + start-state copy per window (copies files and prints the launch commands; starts nothing)
python training/prepare_instances.py --rom /path/to/emerald.gba --n 4 --state /path/to/start.ss1 --out instances

# 2. Start the 4 windows with the printed commands, for example:
flatpak run io.mgba.mGBA -t instances/env0/start.ss1 instances/env0/<rom>
flatpak run io.mgba.mGBA -t instances/env1/start.ss1 instances/env1/<rom>   # ... env2, env3
#    If mGBA can't read the folder (Flatpak sandbox), allow it once:
flatpak override --user --filesystem=$PWD/instances io.mgba.mGBA

# 3. In each window: Tools -> Scripting -> Load script -> games/battle_bridge.lua (first window first).
#    Windows take ports 8888, 8890, 8892, 8894 in the order they load the script.

# 4. Check that all 4 bridges answer (read-only)
python training/check_ports.py 4

# 5. Train on all 4 with one shared model
POKE_N_ENVS=4 python training/train_dqn.py

# Optional: a custom total update size / run length / port list
POKE_N_ENVS=4 POKE_N_STEPS=512 POKE_TOTAL_STEPS=500000 python training/train_dqn.py
POKE_N_ENVS=4 POKE_PORTS=8888,8890,8892,8894 python training/train_dqn.py

# Watch training / summarise logs
tensorboard --logdir ppo_tensorboard
python training/log_report.py

# Stop safely: Ctrl+C in the training terminal (the model is saved before exit)
```

## How the model is saved while training

All saving happens in the main process, in `train_dqn.py` callbacks, and the workers are never involved:

| What | When | Where |
|---|---|---|
| `SafetyCheckpointCallback` | every `max(256 // N_ENVS, 8)` vec-steps (~256 samples) | overwrites `backups/latest_safety.zip` **and** the main file `checkpoints/current/pokemon_ppo_masked_v9.zip` |
| `CheckpointCallback` | once per PPO update (`n_steps`) | numbered milestones `backups/ppo_v9_<steps>_steps.zip`, which accumulate |
| `finally:` block | finish, Ctrl+C or a crash in `learn()` | one last save of the main file |

The main-file save is atomic (`save_atomic`): the zip is written to `<name>.tmp.zip` and then `os.replace`d onto the
real name, so a crash during a write can never leave a half-written model. The next run loads that same main
file (falling back to the v8 baseline only if it doesn't exist), so killing the run at any time loses at most
a few hundred samples. Saved weights are whatever the policy is at that moment, which can be mid-rollout and
between updates, and the file always loads and resumes normally.

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
