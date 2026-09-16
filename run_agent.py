from env import PokemonBattleEnv 

env = PokemonBattleEnv()

episode = 0
while True:
    print(f"\n=== Waiting for battle #{episode} ===")
    obs, info = env.reset()
    print("Battle started! Initial obs:", obs)

    done = False 
    while not done:
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        print(f" action={action} obs={obs} reward={reward}")
        done = terminated or truncated

    print(f"Battle #{episode} finsihed")
    episode += 1
