# HIRO PyTorch Reproduction

This folder keeps the official TensorFlow 1 implementation under
`official_models/research/efficient-hrl` and adds a PyTorch implementation in
`hiro_torch`.

The PyTorch code follows the original framework:

- lower-level UVF-style TD3 policy: input is `(state, relative_goal)`;
- higher-level TD3 policy: outputs a relative goal every `c = 10` steps;
- lower-level reward: `-||s_t[goal_dims] + g_t - s_{t+1}[goal_dims]||`;
- goal transition: `g_{t+1} = g_t - (s_{t+1}[goal_dims] - s_t[goal_dims])`;
- higher-level off-policy correction: candidates are original goal, achieved
  delta, and Gaussian samples around achieved delta; the chosen goal maximizes
  the likelihood of past low-level actions under the current low-level policy.

## Current Status

The core HIRO algorithm is implemented in PyTorch and can be smoke-tested on a
pure NumPy `PointMaze` environment:

```powershell
conda run -n pytorch-cuda121 python scripts/train_torch.py --env PointMaze --steps 1000
```

Outputs are written to `runs/point_maze` by default:

- `metrics.csv`
- `checkpoint_final.pt`
- periodic `checkpoint_*.pt` files when enabled

## Files

- `hiro_torch/networks.py`: TD3 actor, twin critic, target updates.
- `hiro_torch/replay.py`: low-level replay and raw high-level sequence replay.
- `hiro_torch/hiro.py`: HIRO agent, intrinsic reward, goal transition, correction.
- `hiro_torch/simple_envs.py`: lightweight PointMaze for local verification.
- `hiro_torch/envs.py`: environment factory.
- `hiro_torch/train.py`: training loop.
- `scripts/train_torch.py`: command-line entry.

## Next Reproduction Step

The AntMaze/AntPush/AntFall environment layer has been ported from the official
`gym + mujoco_py + tf_agents` stack to modern `gymnasium + mujoco`.

Quick checks:

```powershell
D:\Edge\miniconda\envs\pytorch-cuda121\python.exe scripts\check_env.py --env AntMaze --steps 5
D:\Edge\miniconda\envs\pytorch-cuda121\python.exe scripts\train_torch.py --env AntMaze --steps 100
```

Recommended local runs:

```powershell
# Paper-like update frequency. Slower, best for final runs.
D:\Edge\miniconda\envs\pytorch-cuda121\python.exe scripts\train_torch.py --env AntMaze --steps 100000 --eval-interval 10000 --checkpoint-interval 50000 --log-dir runs\antmaze_100k --high-batch-size 64 --correction-candidates 8

# Faster exploration run. Low-level updates are every 4 env steps; high-level
# HIRO updates still happen on 10-step high-level boundaries. The high-level
# correction batch and candidate count are reduced for speed.
D:\Edge\miniconda\envs\pytorch-cuda121\python.exe scripts\train_torch.py --env AntMaze --steps 100000 --eval-interval 10000 --checkpoint-interval 50000 --log-dir runs\antmaze_100k_fast --train-every 4 --high-batch-size 32 --correction-candidates 4
```

Resume training:

```powershell
# Save replay buffers so the next run can continue from the same training state.
# These checkpoints are much larger than model-only checkpoints.
D:\Edge\miniconda\envs\pytorch-cuda121\python.exe scripts\train_torch.py --env AntMaze --steps 300000 --eval-interval 10000 --checkpoint-interval 100000 --log-dir runs\antmaze_resume --train-every 4 --save-replay

# Continue from 300k to 1M. The --steps value is the final target step, not the
# number of additional steps.
D:\Edge\miniconda\envs\pytorch-cuda121\python.exe scripts\train_torch.py --env AntMaze --steps 1000000 --eval-interval 10000 --checkpoint-interval 100000 --log-dir runs\antmaze_resume --train-every 4 --save-replay --resume runs\antmaze_resume\checkpoint_300000.pt
```

If speed and disk use matter more than exact continuation, omit `--save-replay`.
The checkpoint will contain only the model and optimizer state.
