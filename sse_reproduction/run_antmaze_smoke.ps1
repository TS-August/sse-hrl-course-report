$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
Set-Location -LiteralPath $root

& $python -u "SSE\main.py" `
  --env_name "AntMaze" `
  --test_env_name "AntMaze" `
  --action_max 30. `
  --max_steps 5 `
  --subgoal_freq 5 `
  --subgoal_scale 12. 12. `
  --subgoal_offset 8. 8. `
  --low_future_step 2 `
  --subgoal_dim 2 `
  --l_action_dim 8 `
  --h_action_dim 2 `
  --n_initial_rollouts 2 `
  --n_graph_node 10 `
  --low_bound_epsilon 10 `
  --gradual_pen 5.0 `
  --cuda_num 0 `
  --seed 1 `
  --high_agent `
  --setting "FIFG" `
  --map_size 24 `
  --store_epoch `
  --n_epochs 1 `
  --n_cycles 1 `
  --n_test_rollouts 1 `
  --initial_sample 10 `
  --batch_size 8 `
  --buffer_size 1000 `
  --log_mode "disabled" `
  --ckpt_name "smoke_antmaze_script"
