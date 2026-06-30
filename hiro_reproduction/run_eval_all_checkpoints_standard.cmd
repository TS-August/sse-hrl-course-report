@echo off
cd /d "E:\毕业设计\paper\复现\HIRO"

"D:\Edge\miniconda\envs\pytorch-cuda121\python.exe" scripts\eval_checkpoints.py ^
  --run-dir runs\antmaze_fixed_collision_10m ^
  --episodes 50 ^
  --target all ^
  --seed 4000 ^
  --device cuda ^
  --success-threshold 5.0 ^
  --max-steps 500 ^
  --skip-existing ^
  --jobs 4 ^
  --out runs\antmaze_fixed_collision_10m\checkpoint_eval_50eps_standard_all.csv

if errorlevel 1 exit /b %errorlevel%

"D:\Edge\miniconda\envs\pytorch-cuda121\python.exe" scripts\plot_success_rate.py ^
  --csv runs\antmaze_fixed_collision_10m\checkpoint_eval_50eps_standard_all.csv ^
  --out runs\antmaze_fixed_collision_10m\success_rate_curve_50eps_standard_all.png
