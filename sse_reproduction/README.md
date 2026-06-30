# SSE
Implementation of "Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning" in pytorch.
The code is based on official implementation of [Breadth-First Exploration on Adaptive Grid for Reinforcement Learning](https://github.com/ml-postech/BEAG).

## Installation 
create conda environment
```
conda env create -n sse -f env.yaml python=3.7
conda activate sse
```
if permission denied,
```
chmod +x ./scripts/*.sh
```


## Experiments
To reproduce our experiments, please run the script provided below\
./scripts/{ENV}.sh {GPU} {SEED}
```
example
./scripts/Reacher.sh 0 1
./scripts/AntMaze.sh 2 3
```

The sciprts 'Random_{ENV}.sh' reproduce the ENV with random goal setting. 
'''
example
./scripts/Random_AntMaze.sh 0 0
'''

## Wandb Setting

If you don't want to use wandb, you can add parser "--log_mode 'disabled'" to the scripts.