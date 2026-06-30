# SSE HRL Course Report

Repository: https://github.com/TS-August/sse-hrl-course-report.git

This repository contains the course report, reproduction code, experiment scripts, selected logs, metrics, and visualization assets for the topic:

`稀疏奖励下的分层强化学习算法研究`

## Repository Layout

| Path | Content |
| --- | --- |
| `report/` | Markdown reports, generated Word report, latest presentation PDF, and oral presentation script |
| `report_assets/` | Figures used by the report |
| `hiro_reproduction/` | Lightweight HIRO PyTorch reproduction code, scripts, selected metrics, and evaluation plots |
| `sse_reproduction/SSE/` | SSE algorithm source code |
| `sse_reproduction/envs/` | AntMaze and AntKeyChest environment code |
| `sse_reproduction/scripts/` | Trace export and visualization helper scripts |
| `sse_reproduction/run_umaze_1m_wandb.ps1` | U-Maze reproduction script |
| `sse_reproduction/run_antkeychest_5m_wandb.ps1` | AntKeyChest reproduction script |
| `sse_reproduction/run_shortcut_detour_700k_wandb.ps1` | Shortcut-Detour reliability experiment script |
| `sse_reproduction/exp/` | Selected `config.json` and `metrics.csv` outputs |
| `sse_reproduction/runs/` | Selected task status, stdout/stderr, and W&B summaries |
| `sse_reproduction/visualizations/` | Selected trajectory CSV/video outputs referenced by the report |

## Experiments

| Run ID | Environment | Purpose |
| --- | --- | --- |
| `antmaze_fixed_collision_10m` | HIRO AntMaze Fixed Collision | HIRO baseline reproduction and checkpoint evaluation |
| `umaze_1m_seed1` | `AntMaze` | Basic sparse-reward maze reproduction |
| `antkeychest_5m_seed1` | `AntMazeKeyChest` | Long-horizon key-and-goal task |
| `shortcut_detour_1m_seed1` | `AntMazeShortcutDetour-v0` | Checks whether planning favors a longer but more reliable path over a shorter unreliable path |

The PowerShell scripts use `PYTHON_EXE` when it is set; otherwise they call `python` from `PATH`.

Example:

```powershell
$env:PYTHON_EXE="<path-to-python.exe>"
.\run_umaze_1m_wandb.ps1 -Gpu 0 -Seeds 1
```

Large checkpoints, TensorBoard event directories, W&B caches, and local runtime caches are intentionally excluded.

Latest presentation files:

- `report/复现工作汇报.md`
- `report/复现工作汇报.pdf`
- `report/复现工作汇报_讲稿.md`
