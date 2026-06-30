GPU=$1
SEED=$2


CUDA_VISIBLE_DEVICES=${GPU} python SSE/main.py \
--env_name 'AntMazeDoubleBottleneck-v0' \
--test_env_name 'AntMazeDoubleBottleneck-eval-v0' \
--action_max 30. \
--max_steps 1200 \
--subgoal_freq 1200 \
--subgoal_scale 12. 20. \
--subgoal_offset 8. 16. \
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
--store_epoch \
--fail_weight 10.0 \
--fail_count 5
