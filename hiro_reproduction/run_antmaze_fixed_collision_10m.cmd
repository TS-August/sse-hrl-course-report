@echo off
cd /d "%~dp0"
if not exist runs\antmaze_fixed_collision_10m mkdir runs\antmaze_fixed_collision_10m
"D:\Edge\miniconda\envs\pytorch-cuda121\python.exe" -u scripts\train_torch.py --env AntMaze --steps 10000000 --eval-interval 50000 --checkpoint-interval 100000 --log-dir runs\antmaze_fixed_collision_10m --high-batch-size 100 --correction-candidates 8 --train-every 1 --save-final-replay --wandb --wandb-project hiro-pytorch --wandb-entity eaijia405-hitsz --wandb-run-name antmaze_fixed_collision_10m_seed0 --wandb-mode online > runs\antmaze_fixed_collision_10m\train_stdout.log 2> runs\antmaze_fixed_collision_10m\train_stderr.log
