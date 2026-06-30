import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rl.launcher import launch
import envs
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='AntMaze')
    parser.add_argument('--test_env_name', type=str, default='AntMaze')
    parser.add_argument('--setting', type=str, default='FIFG', choices=['FIFG', 'FIRG'])
    parser.add_argument('--action_max', type=float, default=30.) #network action_max > always 1
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--subgoal_freq', type=int, default=40)
    parser.add_argument('--subgoal_scale', type=float, nargs='+', default=[12., 12.])
    parser.add_argument('--subgoal_offset', type=float, nargs='+', default=[8., 8.])
    parser.add_argument('--low_future_step', type=int, default=150)
    parser.add_argument('--init_dist', type=float, default=2.)
    
    parser.add_argument('--subgoal_dim', type=int, default=2)
    parser.add_argument('--l_action_dim', type=int, default=8)
    parser.add_argument('--h_action_dim', type=int, default=2)
    parser.add_argument('--cutoff', type=float, default=30)
    parser.add_argument('--n_initial_rollouts', type=int, default=200) 
    parser.add_argument('--absolute_goal', action="store_true")

    parser.add_argument('--n_graph_node', type=int, default=300)
    parser.add_argument('--low_bound_epsilon', type=int, default=10)
    parser.add_argument('--gradual_pen', type=float, default= 5.0)

    ################################################################################################

    parser.add_argument('--low_future_p', type=float, default=0.8)
    parser.add_argument('--low_future_p_g', type=float, default=1.1)
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--clip_return', type=float, default=80) 
    parser.add_argument('--start_planning_epoch', type=int, default=0)
    parser.add_argument('--subgoaltest_p', type=float, default=0.2)

    
    #wandb project name
    parser.add_argument('--project_name', type=str, default='SSE')
    parser.add_argument('--log_mode', type=str, default='online')

    #cuda
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--cuda_num', type=int, default=0)

    #directory
    parser.add_argument('--save_dir', type=str, default='exp/')
    parser.add_argument('--ckpt_name', type=str, default='')
    parser.add_argument('--resume_ckpt', type=str, default='')
    parser.add_argument('--store_epoch', action='store_true')

    # Load model
    parser.add_argument('--load_ckpt_name', type=str, default=None)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--low_agent', action='store_true')
    parser.add_argument('--high_agent', action='store_true')
    parser.add_argument('--low_replay', action='store_true')
    parser.add_argument('--high_replay', action='store_true')

    #network and training
    parser.add_argument('--use_reverse_dist_func', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=9999)
    parser.add_argument('--target_env_steps', type=int, default=0)
    parser.add_argument('--n_cycles', type=int, default=15)
    parser.add_argument('--high_optimize_freq', type=int, default=50)
    parser.add_argument('--densify_freq', type=int, default=1)

    parser.add_argument('--n_batches', type=int, default=1)
    parser.add_argument('--hid_size', type=int, default=256)
    parser.add_argument('--n_hids', type=int, default=3)
    parser.add_argument('--activ', type=str, default='relu')
    parser.add_argument('--noise_eps', type=float, default=0.1)
    
    
    parser.add_argument('--random_eps', type=float, default=0.2)
    parser.add_argument('--buffer_size', type=int, default=2500000)
    parser.add_argument('--gamma_low', type=float, default=0.99)
    parser.add_argument('--gamma_high', type=float, default=0.4)
    
    parser.add_argument('--action_l2', type=float, default=0.01)
    parser.add_argument('--lr_actor_low', type=float, default=0.0001)
    parser.add_argument('--lr_critic_low', type=float, default=0.001)
    parser.add_argument('--lr_actor_high', type=float, default=0.000005)
    parser.add_argument('--lr_critic_high', type=float, default=0.00005)
    parser.add_argument('--polyak', type=float, default=0.995)

    parser.add_argument('--target_update_freq', type=int, default=10)
    parser.add_argument('--actor_update_freq', type=int, default=2)
    parser.add_argument('--grad_norm_clipping', type=float, default=-1.0)
    parser.add_argument('--grad_value_clipping', type=float, default=-1.0)

    #test
    parser.add_argument('--n_test_rollouts', type=int, default=20)
    parser.add_argument('--eval_render', type=bool, default=False)

    #graph construct
    parser.add_argument('--q_offset', action='store_true')
    parser.add_argument('--initial_sample', type=int, default=6000)
    parser.add_argument('--use_oracle_G', type=bool, default=False)
    parser.add_argument('--graph_threshold', type=float, default=2.0)

    parser.add_argument('--real_dist_weight', type=float, default=0.5)
    parser.add_argument('--agent_dist_weight', type=float, default=0.5)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--video_freq', type=int, default=10)
    parser.add_argument('--no_state', action='store_true')
    parser.add_argument('--fail_weight', type=float, default=5.0)
    parser.add_argument('--move_count', type=int, default=500)

    parser.add_argument('--grad_norm_clipping_high', type=float, default=0.5)
    parser.add_argument('--grad_value_clipping_high', type=float, default=2.0)
    parser.add_argument('--algorithm_name', type=str, default='SSE')
    parser.add_argument('--grid_size', type=float, default=2.)
    parser.add_argument('--map_size', type=int, default=48)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--exp_rate', type=float, default=0.2)
    parser.add_argument('--detour', action='store_true')
    parser.add_argument('--fail_count', type=int, default=3)
    
    parser.add_argument('--offset', type=float, default=4.0)
    parser.add_argument('--inside_count', type=int, default=100)
    parser.add_argument('--lambda1_inside', type=float, default=2.0)
    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--epoch_save_iter', type=int, default=300)
    
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    import os
    algo = launch(args)
    algo.run()
