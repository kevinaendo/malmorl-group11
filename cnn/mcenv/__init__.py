from gym.envs.registration import register

register(
    id='MinecraftEnv-v0',
    entry_point='mcenv.minecraft_env:MinecraftEnv',
	kwargs={'mission_file': 'pg_gym.xml'},
)
