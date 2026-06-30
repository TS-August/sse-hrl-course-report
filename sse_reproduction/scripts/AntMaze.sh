GPU=$1
SEED=$2

CUDA_VISIBLE_DEVICES=${GPU} python SSE/main.py \
--env_name 'AntMaze' \
--test_env_name 'AntMaze' \
--action_max 30. \
--max_steps 600 \
--subgoal_freq 600 \
--subgoal_scale 12. 12. \
--subgoal_offset 8. 8. \
--low_future_step 150 \
--subgoal_dim 2 \
--l_action_dim 8 \
--h_action_dim 2 \
--n_initial_rollouts 200 \
--n_graph_node 300 \
--low_bound_epsilon 10 \
--gradual_pen 5.0 \
--cuda_num 0 \
--seed ${SEED} \
--high_agent \
--setting 'FIFG' \
--map_size 24 \
--store_epoch
