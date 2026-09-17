from env import PokemonBattleEnv 
from logger import init_csv, log_step, log_battle_end 

env = PokemonBattleEnv()
init_csv()

episode = 0
while True:
    print(f"\n=== Waiting for battle #{episode} ===")
    obs, info = env.reset()
    print("Battle started! Initial obs:", obs)

    done = False 
    step = 0
    player_lost = False 
    while not done:
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        print(f" action={action} obs={obs} reward={reward}")

        log_step(episode, step, action, obs, reward, terminated, player_lost)

        done = terminated or truncated
        step += 1

    won = reward > 0
    log_battle_end(episode, won)
    print(f"Battle #{episode} finished. Result: {'WIN' if won else 'LOSS'}")
    episode += 1
