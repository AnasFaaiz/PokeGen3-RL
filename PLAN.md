# Plan: making the model a better trainer

Everything below is written and passes the offline tests (`training/test_*.py`, the Lua harness, `luac -p`). **Nothing has
been run against the live game and nothing is committed.** The v8 training run holds the emulator bridge, so all live checks
wait for it to finish.

## What exists

| Area | What | Where |
|---|---|---|
| Reward | fractional-XP level term, per-gap difficulty baseline (terminal shift), flee penalty, optional difficulty-scaled HP, optional lower win bonus | `training/env.py` |
| Data for harder fights | swap gap 12 -> 6 with hysteresis and a keep-lead chance, damaged starts (heal threshold 0.2); catch reward "useful" band follows the area level | `swap_policy.py`, `env.py` |
| Understanding its mons | 60 combat features (expected damage, KO, speed, stats, bench, priority, heal/status/setup tags) | `damage_calc.py`, `move_tags.py`, `env.py` |
| Opponent trainers | decomp tables (855 trainers, 552 trainer objects), 8 team features, `TRAINER` / `TRAINERSCAN` / `TRAINERFLAGS` bridge commands | `trainer_info.py`, `games/emerald/data_gen/generate_trainers.py` |
| **Move teaching** | scorer + planner from level-up learnsets, Lua `MONMOVES` / `SETMOVES`, 4 model-chosen actions, shaping reward | `move_policy.py`, `mon_struct.py`, `inventory.lua`, `env.py` |
| Measurement | `winrate_by_gap`, `winrate_all_by_gap`, `flee_rate_by_gap`, `turns/return_by_gap`, `winrate_trainer`, `analysis/run_eval.py` (`--stochastic`), `training/log_report.py` | `train_dqn.py`, `logger.py` |
| Speed | PPO update every 512 samples with a KL stop, `profile.csv` timing log, parallel emulators (`POKE_N_ENVS`), `check_ports.py`, `prepare_instances.py` | `train_dqn.py`, `check_ports.py`, `prepare_instances.py`, `games/bridge_ports.lua` |
| Logs | run id in every row + `logs/runs.csv`, compact observations, size rotation, species/map columns, `moves.csv`, correct outcome names | `logger.py` |

### Move teaching in one paragraph
The bridge answers the game's in-battle "delete a move to make room?" prompt with No, so a mon with four moves never learned
anything. Now, with `POKE_MOVE_TEACH=1`, at the first decision of a battle the env scans the party (`MONMOVES`), picks the mon
with the biggest worthwhile gain and offers its best level-up candidates (scored from the mon's own Attack / Sp. Atk, STAB,
accuracy, quirks such as two-turn moves, role value for heal/status/setup, type coverage; HM moves are never overwritten or
taught, at least two damaging moves are kept). The model picks one with actions 18-21 (no game turn, no turn cost, small
shaping reward from the score gain). The edit is queued by the mon's personality value and written between battles in the
overworld (`SETMOVES`), which refuses on a pid mismatch, a bad checksum or a battle, rolls back if the read-back differs, and
clears the replaced slot's PP Ups like the game does. If the moves changed during the battle (a level-up learned one) the plan
is redone on the current set. **The shaping reward is paid only after a verified edit** (with the next battle's first step, and
kept out of the logged return); a refused edit costs a small penalty and three failures in a row switch teaching off for the run,
so a broken write path can't be farmed for reward.
Caveats: a teach pays off many battles later, so shaping gives the early signal; the model chooses only among the scorer's top
four, so the scorer bounds quality; the real game's struct layout is verified only by the live checklist below.

## Flags and environment variables

| Variable | Effect | Default |
|---|---|---|
| `POKE_LEGACY_OBS=1` | 431-value observation (v8's input); also disables teaching | off |
| `POKE_LEGACY_MOVE_TYPES=1` | the old (wrong) move-type numbering and empty-slot type, to evaluate v8 as it was trained | off |
| `POKE_FLEE_PENALTY=40` | charge for the player's own run/teleport in a wild battle (a flee returns about -60); `0` = free flee | 40 |
| `POKE_DIFF_HP=1` | scale HP-lost and HP-kept terms by 1/(1+gap/10) | off |
| `POKE_WIN_REWARD=60` | win bonus (loss stays -100); at 60 a full win only just beats the best catch (132 vs 125) | 100 |
| `POKE_ENT_ANNEAL=1` | lower `ent_coef` 0.01 -> 0.002 over the run (`train/ent_coef_sched`) | off |
| `POKE_FRAME_STACK=1` | append last turn's 60 combat features + last action (+61 inputs) | off |
| `POKE_MOVE_TEACH=1` | 4 teach actions (22 total) + 44 teach inputs; `POKE_TEACH_SHAPING` (default 4) scales its reward | off |
| `POKE_N_ENVS=N` | N parallel emulator windows (ports 8888, 8890, ... via `POKE_PORT_BASE` / `POKE_PORT_STEP`, or an explicit `POKE_PORTS=...`) | 1 |
| `POKE_N_STEPS` / `POKE_BATCH_SIZE` / `POKE_N_EPOCHS` / `POKE_TARGET_KL` | PPO update: total samples per update across all envs, minibatch, epochs, KL early stop. v8 used 2048 / 512 | 512 / 128 / 10 / 0.03 |
| `POKE_LOG_OBS_EVERY=N` | keep the observation on every Nth per-step log row (default every row) | 1 |
| `POKE_LOG_CSV_MAX_MB`, `POKE_LOG_TXT_MAX_MB` | rotate `battle_data.csv` / `battle_log.txt` past this size (keeps 2 old files) | 200 / 20 |

Observation size: 431 (legacy) / 499 (default) / +61 stack / +44 teach (604 with both). Model files: `pokemon_ppo_masked_v9[_stack][_teach]`.
Build one from v8 with `python training/expand_model_v9.py [--stack] [--teach]` (new inputs start at zero; teach actions start rare,
bias -2; checked to behave like v8 on the old actions).

## Live-test checklist (run after the v8 training run finishes and the Lua script is reloaded in mGBA)

1. `python training/bridge_cli.py`, then `INVENTORY`: the `party progress:` line shows fractional levels whose floor equals `lv=`.
2. Baseline: `POKE_LEGACY_OBS=1 POKE_LEGACY_MOVE_TYPES=1 python analysis/run_eval.py --model checkpoints/current/pokemon_ppo_masked_v8 --battles 100`
   (add `--stochastic` for the sampling policy). Rows go to `logs/eval_results.csv`.
3. Trainer ids: `TRAINER` replies `TRAINER:ERR:no_address`. In a trainer fight send `TRAINERSCAN:<id>` (id from
   `game_data/trainers.json`), repeat in a second fight, put the address that matched both in `emerald_config.lua`
   (`INV.TRAINER_OPPONENT_ADDR`). `TRAINERFLAGS:<id>` should be 1 for a beaten trainer, 0 otherwise (else the flags offset 0x1270 is wrong).
4. **Move teaching** (only if you want it): `MONMOVES:0` .. `MONMOVES:5` must all end in `:1` (checksum_ok). Then
   `SETMOVES:<slot>:<pid>:<m1>:<m2>:<m3>:<m4>:<pp1>:<pp2>:<pp3>:<pp4>:dry` on one mon must reply `SETMOVES:DRY:ok`; then without `dry`
   on a mon you don't mind changing, and check the moves in the game's summary screen.
5. Train: `python training/expand_model_v9.py` then `python training/train_dqn.py` (or with the flags above). One flag per run so
   results stay attributable. `python training/log_report.py` summarises the logs per run, per level gap and per enemy species.
6. Compare: `python analysis/run_eval.py --model checkpoints/current/pokemon_ppo_masked_v9 --battles 100` against the baseline row.
7. Commit in pieces once 1-2 pass (the unrelated modified `checkpoints/current/pokemon_ppo_masked_v8.zip` stays out).

Stop the v8 run before step 1: the bridge accepts one client, and the running process still writes the old 15-column
`battles.csv` (the new process moves old-header logs aside automatically).

## What is left (pending report)

**Code-complete offline: 7 of the 7 planned work areas** (reward, harder data, combat features, trainer features, move teaching, logs, tests).
**Still pending: 8 items.** 4 are live verification (1-3 and the first v9 run, listed first), 3 are optional offline builds that need a
decision or v9 results, 1 is a live-dependent refinement. Nothing pending is a known bug.

| Item | Needs |
|---|---|
| Live checks above, baseline eval, first v9 run, commits | v8 run finished, your emulator |
| Trainer-ID RAM address and `TRAINERFLAGS` layout | a trainer fight |
| Move teaching on the real game (checksum, PP, summary screen) | step 4 above |
| Teaching sources beyond level-up (TM/HM compatibility, tutors, egg moves) and ability-aware scoring (Guts, Technician, Levitate...) | decision after teaching works |
| Embeddings for species/moves and a recurrent policy | v9 results (needs distillation or a new model) |
| Navigation toward undefeated trainers (helpers exist in `trainer_info.py`, nothing calls them) | after trainer ids work |
| Exact own stats from the party struct instead of IV-15 estimates | slot-order mapping check live |
| Phase 3 reward experiments (`POKE_WIN_REWARD`, `POKE_DIFF_HP`) | v9 baseline first, one per run |

## Findings that shaped the design (from `logs/battles.csv`, ~15k wild battles)
- Wins were ~99% of decisive fights, so the win/loss signal was nearly saturated; the remaining signal is HP kept, turns and growth.
- **Fleeing was free and invisible**: teleport rate 9.1% at level gap <= 0 vs 0.6% above 20, a flee returned about -22 vs -274 for a loss,
  and no win rate counted it. Fixed with a penalty and flee-aware metrics.
- **Catching is the model's own THROW action** (`SCRIPTED_CATCH = False`). The old scripted catch rule is dormant; I made it stock-driven
  anyway so it would not stall under the smaller swap gap if it is ever switched back on, and the catch reward's "useful" band now follows
  the area level. Balls cannot be rebought (money 0), so catches (43 balls in the bag at last check: 35 Poke, 7 Premier, 1 Great) limit how many swaps the box can supply.
- **Move types in the observation were mis-numbered** (Fire..Dark moves) and empty move slots looked like Dark moves: fixed.
- The logs had no run id, 216 MB of full-precision observations and mislabelled outcomes: fixed (see "What exists").

## Sample speed (why learning looked flat) and parallel training

The v8 run did ~1,030 steps/hour (6.7 steps and ~23.5 s per battle: ~12.3 s fighting, ~11.2 s walking, healing and waiting), so a
2,048-step update took ~2 hours, 26,624 total steps were only 13 updates, and explained variance stayed ~0. 500k steps would take
~20 days on one emulator. Three levers, cheapest first:

1. **Make one emulator faster (no code).** The config has `fastForwardRatio=8` but the run looked like ~1x (34% CPU). Turn on fast
   forward (Tab) or set the ratio to unbounded (Settings -> Emulation), and in the game's Options set BATTLE SCENE: OFF and
   TEXT SPEED: FAST. `logs/profile.csv` (written every 25 battles from now on) and `python training/log_report.py` show where the
   time goes (walking, bridge commands, waiting).
2. **Update more often.** Default is now 512 samples per update (batch 128, KL early stop): about 4x more updates from the same data.
3. **Parallel emulators: about N x steps per hour.** Each mGBA window is one environment with its own save, on its own bridge port.

### Parallel training how-to (what you do)
1. In the current window, standing on the grass route, save a state (File -> Save state -> slot 1).
2. `python training/prepare_instances.py --rom <Emerald .gba> --n 4 --state <that .ss1>` copies the ROM, its `.sav` and the state into
   `instances/env0..3/` (one save file per window; `instances/` is git-ignored) and prints one launch command per window.
3. Run those commands (one Flatpak window each; keep them open). If a window cannot read or save there, run the `flatpak override`
   command the script prints.
4. In each window: Tools -> Scripting -> Load script -> `games/battle_bridge.lua`, **first window first**. The console prints the port
   it took: the first window gets 8888/8889, the second 8890/8891, and so on (`games/bridge_ports.lua` picks the first free one).
   Reload the script in every window whenever it changes.
5. `python training/check_ports.py 4`: every env should say OK, with the map and `new-lua=yes`.
6. `POKE_N_ENVS=4 python training/train_dqn.py`. Rows in the logs carry `..._e0` .. `_e3`; episode numbers are `i * 10,000,000 + n`,
   so they never collide. The per-update checkpoint and `backups/latest_safety.zip` are written by the main process.
Limits and failure modes: each window uses ~1 core and ~650 MB (4 fit easily on 16 cores / 22 GB); every window has its own party, PC box
and balls; if a window dies training stops with a clear connection error (no silent reconnect: it could resume from a different game
state) and you restart it from the last checkpoint.
