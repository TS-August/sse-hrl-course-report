GPU=$1
SEED=$2



CUDA_VISIBLE_DEVICES=${GPU} python SSE/main.py \
--env_name 'AntMazeComplex-v0' \
--test_env_name 'AntMazeComplex-v0' \
--action_max 30. \
--max_steps 2000 \
--subgoal_freq 2000 \
--subgoal_scale 28. 28. \
--subgoal_offset 24. 24. \
--low_future_step 150 \
--subgoal_dim 2 \
--l_action_dim 8 \
--h_action_dim 2 \
--n_initial_rollouts 700 \
--n_graph_node 500 \
--low_bound_epsilon 10 \
--gradual_pen 5.0 \
--cuda_num 0 \
--seed ${SEED} \
--high_agent \
--setting FIFG \
--map_size 56 \
--store_epoch