from env import PokemonBattleEnv

env = PokemonBattleEnv()

episode = 0
while True:
    print(f"\n=== Waiting for battle #{episode} ===")
    obs, info = env.reset()
    print("Battle started! Initial obs:", obs)

    done = False
    step = 0
    while not done:
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        player_lost = info.get("player_lost", False)
        print(f" action={action} obs={obs} reward={reward}")

        done = terminated or truncated
        step += 1

    won = reward > 0
    print(f"Battle #{episode} finished. Result: {'WIN' if won else 'LOSS'}")
    episode += 1
