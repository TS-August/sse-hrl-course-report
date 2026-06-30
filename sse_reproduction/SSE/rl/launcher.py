import gym
import random
import numpy as np
import torch
import time
import os.path as osp
import wandb
from rl.utils.run_utils import Monitor
from rl.replay.planner import LowReplay, HighReplay
from rl.learn.sse import HighLearner, LowLearner
from rl.agent.agent import LowAgent, HighAgent
from rl.algo.sse import Algo
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from envs.antenv import EnvWithGoal, GatherEnv, EnvWithKeyChest, EnvWithDoubleKeyChest
from envs.antenv.create_maze_env import create_maze_env
from envs.antenv.create_gather_env import create_gather_env
from envs.antenv.ant_maze_shortcut_detour import AntMazeShortcutDetourEnv, AntMazeShortcutDetourEvalEnv
import datetime

def get_env_params(env, args):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'sub_goal': args.subgoal_dim,
              'l_action_dim': args.l_action_dim,
              'h_action_dim': args.h_action_dim,
              'action_max': args.action_max,
              'max_timesteps': args.max_steps}
    return params


def launch(args):
    now = datetime.datetime.now()
    name = args.ckpt_name if len(args.ckpt_name) > 0 else 'SSE'

    wandb.init(project=args.project_name, group=args.env_name, name=name, config=vars(args), sync_tensorboard=False, mode= args.log_mode, settings=wandb.Settings(disable_code=True, _disable_stats=True))
        
    wandb.define_metric('Total Timesteps')

    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        test_env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    elif args.env_name in ["AntMaze", "AntMazeComplex-v0", "AntMazeP"]:
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        env.setting = args.setting
        test_env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    elif args.env_name in ['AntMazeKeyChest']:
        env = EnvWithKeyChest(create_maze_env(args.env_name, args.seed), args.env_name)
        env.setting = args.setting
        test_env = EnvWithKeyChest(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    elif args.env_name in ['AntMazeDoubleKeyChest']:
        env = EnvWithDoubleKeyChest(create_maze_env(args.env_name, args.seed), args.env_name)
        env.setting = args.setting
        test_env = EnvWithDoubleKeyChest(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    elif args.env_name in ['AntMazeShortcutDetour-v0', 'AntMazeShortcutDetour']:
        env = AntMazeShortcutDetourEnv(seed=args.seed)
        env.setting = args.setting
        test_env = AntMazeShortcutDetourEvalEnv(seed=args.seed)
        test_env.evaluate = True
    else:
        env = gym.make(args.env_name)
        test_env = gym.make(args.test_env_name)
        if args.env_name == "Reacher3D-v0":
            test_env.evaluate = True
        try:
            env.setting = args.setting
        except:
            pass
    seed = args.seed

    env.seed(seed)
    test_env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    assert np.all(env.action_space.high == -env.action_space.low)
    env_params = get_env_params(env, args)
    low_reward_func = env.low_reward_func
    high_reward_func = env.high_reward_func
    monitor = Monitor(args.max_steps)


    ckpt_name = args.ckpt_name
    if len(ckpt_name) == 0:
        data_time = time.ctime().split()[1:4]
        ckpt_name = data_time[0] + '-' + data_time[1]
        time_list = np.array([float(i) for i in data_time[2].split(':')], dtype=np.float32)
        for time_ in time_list:
            ckpt_name += '-' + str(int(time_))
        args.ckpt_name = ckpt_name
    
    low_agent = LowAgent(env_params, args)
    high_agent = HighAgent(env_params, args)


    low_replay = LowReplay(env_params, args, low_reward_func)
    high_replay = HighReplay(env_params, args, high_reward_func, monitor, high_agent)
    low_learner = LowLearner(low_agent, monitor, args)
    high_learner = HighLearner(high_agent, monitor, args)

    algo = Algo(
        env=env, env_params=env_params, args=args,
        test_env=test_env, 
        low_agent=low_agent, high_agent = high_agent, low_replay=low_replay, high_replay=high_replay, monitor=monitor, 
        low_learner=low_learner, high_learner=high_learner,
        low_reward_func=low_reward_func, high_reward_func=high_reward_func
    )
    return algo
