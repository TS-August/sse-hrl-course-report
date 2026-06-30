param(
  [int]$Gpu = 0,
  [int[]]$Seeds = @(1),
  [string]$Project = "SSE",
  [string]$RunSuffix = "",
  [int]$TargetTimesteps = 700000
)

$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$targetTimesteps = $TargetTimesteps
$runLabel = if ($targetTimesteps -eq 1000000) { "1m" } else { "$([int]($targetTimesteps / 1000))k" }

Set-Location -LiteralPath $root

$netrc = Join-Path $env:USERPROFILE "_netrc"
$hasApiKey = $env:WANDB_API_KEY -or (Test-Path -LiteralPath $netrc)
if (-not $hasApiKey) {
  throw "W&B is not logged in. Run: `"$python`" -m wandb login <YOUR_API_KEY>, then rerun this script."
}

foreach ($seed in $Seeds) {
  $ckptName = "shortcut_detour_${runLabel}_seed$seed$RunSuffix"
  $runDir = Join-Path $root "runs\$ckptName"
  $stdout = Join-Path $runDir "train_stdout.log"
  $stderr = Join-Path $runDir "train_stderr.log"
  $status = Join-Path $runDir "task_status.txt"
  $errorLog = Join-Path $runDir "task_error.log"

  New-Item -ItemType Directory -Force -Path $runDir | Out-Null

  $env:CUDA_VISIBLE_DEVICES = "$Gpu"
  $env:WANDB_MODE = "online"
  $env:WANDB_DIR = $runDir
  $env:WANDB_CACHE_DIR = Join-Path $runDir "wandb_cache"
  $env:WANDB_DATA_DIR = Join-Path $runDir "wandb_data"
  $env:WANDB_PROJECT = $Project

  try {
    "started $(Get-Date -Format o), seed=$seed, target_training_timesteps=$targetTimesteps" | Set-Content -LiteralPath $status -Encoding UTF8
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python -u "SSE\main.py" `
      --env_name "AntMazeShortcutDetour-v0" `
      --test_env_name "AntMazeShortcutDetour-eval-v0" `
      --action_max 30. `
      --max_steps 500 `
      --subgoal_freq 250 `
      --subgoal_scale 12. 8. `
      --subgoal_offset 10. 6. `
      --low_future_step 100 `
      --subgoal_dim 2 `
      --l_action_dim 8 `
      --h_action_dim 2 `
      --n_initial_rollouts 150 `
      --n_graph_node 120 `
      --initial_sample 4000 `
      --low_bound_epsilon 8 `
      --gradual_pen 5.0 `
      --cuda_num $Gpu `
      --seed $seed `
      --high_agent `
      --setting "FIFG" `
      --map_size 28 `
      --offset 4 `
      --grid_size 2.0 `
      --store_epoch `
      --epoch_save_iter 10 `
      --epsilon_min 0.1 `
      --gamma_high 0.4 `
      --fail_weight 5.0 `
      --project_name $Project `
      --log_mode "online" `
      --ckpt_name $ckptName `
      --target_env_steps $targetTimesteps `
      1> $stdout 2> $stderr
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) {
      throw "Training exited with code $exitCode. See $stderr"
    }
    "finished $(Get-Date -Format o)" | Add-Content -LiteralPath $status -Encoding UTF8
  }
  catch {
    $_ | Out-String | Set-Content -LiteralPath $errorLog -Encoding UTF8
    throw
  }
}
