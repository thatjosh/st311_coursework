# ST311 Project

This repository contains training and testing scripts for building an RL agent under PPO policy and under various memory architectures (LSTM, Mamba, and Transformer) in the MiniGrid environment.

This repository extended the RL Starter Files repository.

## Installation

1. Clone this repository.

2. Install `minigrid` environments and `torch-ac` RL algorithms:

```
pip3 install -r requirements.txt
```

**Note:** To modify `torch-ac` algorithms, you will need to rather install a cloned version, i.e.:

```
git clone https://github.com/lcswillems/torch-ac.git
cd torch-ac
pip3 install -e .
```

## Example of use

The following will kick start the script to train an agent under the Red Blue Door env in MiniGrid.

```
python3 -m scripts.train --algo ppo --env MiniGrid-RedBlueDoors-6x6-v0 --model RedBlueDoors --recurrence 4 --save-interval 10 --frames 1000000
```

<p align="center"><img src="README-rsrc/visualize-redbluedoors.gif"></p>

## Files

This package contains:

- scripts to:
  - train an agent \
    in `script/train.py`
  - test agent's behavior \
    in `script/test.py`
  - run a hyperparameter grid search \
    in `grid_search.py`
- a default agent's model \
  in `model.py`
- LSTM, Mamba, and Transformer model architecture \
  in `memory_modules.py`
- utilitarian classes and functions used by the scripts \
  in `utils`
- code to:
  - analyse results from hyperparam tuning \
    in `regression_analysis.Rmd`
