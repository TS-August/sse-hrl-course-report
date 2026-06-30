from gym.envs.registration import registry, register, make, spec

register(
        id='AntMazeBottleneck-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeBottleneckEnv',
        max_episode_steps=600,
        reward_threshold=0.0,
    )

register(
        id='AntMazeBottleneck-eval-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeBottleneckEvalEnv',
        max_episode_steps=600,
        reward_threshold=0.0,
    )

register(
        id='AntMazeDoubleBottleneck-v0',
        entry_point='envs.antenv.ant_maze_double_bottleneck:AntMazeDoubleBottleneckEnv',
        max_episode_steps=1200,
        reward_threshold=0.0,
    )

register(
        id='AntMazeDoubleBottleneck-eval-v0',
        entry_point='envs.antenv.ant_maze_double_bottleneck:AntMazeDoubleBottleneckEvalEnv',
        max_episode_steps=1200,
        reward_threshold=0.0,
    )

register(
        id='AntMazeShortcutDetour-v0',
        entry_point='envs.antenv.ant_maze_shortcut_detour:AntMazeShortcutDetourEnv',
        max_episode_steps=500,
        reward_threshold=0.0,
    )

register(
        id='AntMazeShortcutDetour-eval-v0',
        entry_point='envs.antenv.ant_maze_shortcut_detour:AntMazeShortcutDetourEvalEnv',
        max_episode_steps=500,
        reward_threshold=0.0,
    )

register(
        id='Reacher3D_wall',
        entry_point='envs.fetchenv.create_fetch_env:create_fetch_env',
        kwargs={'env_name': 'Reacher3D_wall'},
        max_episode_steps=100
)
register(
        id='Reacher3D_DoubleGoal',
        entry_point='envs.fetchenv.create_DoubleGoal_env:create_DoubleGoal_env',
        kwargs={'env_name': 'Reacher3D_Doublegoal'},
        max_episode_steps=100
)
