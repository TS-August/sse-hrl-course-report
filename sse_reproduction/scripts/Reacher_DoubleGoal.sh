GPU=$1
SEED=$2

CUDA_VISIBLE_DEVICES=${GPU} python SSE/main.py \
--env_name 'Reacher3D_DoubleGoal' \
--test_env_name 'Reacher3D_DoubleGoal' \
--action_max 20. \
--max_steps 200 \
--subgoal_freq 200 \
--subgoal_scale 1. 1. 1. \
--subgoal_offset 0. 0. 0. \
--low_future_step 100 \
--subgoal_dim 3 \
--l_action_dim 7 \
--h_action_dim 3 \
--cutoff 10 \
--n_initial_rollouts 200 \
--n_graph_node 300 \
--low_bound_epsilon 5 \
--gradual_pen 0.0 \
--cuda_num 0 \
--seed ${SEED} \
--high_agent \
--exp_rate 0.2 \
--eta 0.2 \
--grid_size 1.0 \
--map_size 2 \
--offset 1 \
--lr_critic_high 0.00005 \
--lr_actor_high 0.000005 \
--epsilon_min 0.1 \
--lambda1 0.1 \
--lambda1_inside 0.15 \
--inside_count 50 \
--absolute_goal

