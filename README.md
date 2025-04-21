# Global Rewards in Multi-Agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems

This software uses multi-agent SAC with a central bipartite matching in combination with credit assignment based on COMA to train and test a policy, represented by a neural network, that dispatches vehicles to requests in an autonomous mobility on demand system.

This method is proposed in:

> Heiko Hoppe, Tobias Enders, Quentin Cappart, Maximilian Schiffer (2023). Global Rewards in Multi-Agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems. Proceedings of the 6th Annual Learning for Dynamics & Control Conference (L4DC), Proceedings of Machine Learning Research (PMLR), voume 242, pp. 260-272.

All components (code, data, etc.) required to run the code for the instances considered in the paper are provided here. This includes the greedy benchmark algorithm.

## Overview
The directory `algorithms` contains:
- The environment implementation in `environment.py`.
- The greedy benchmark algorithm in `greedy.py`, which can be executed using `main_greedy.py` with arguments as the exemplary ones in `args_greedy_XX_small/large_zones.txt` (see comments in `main_greedy.py` for explanations of the arguments).
- The remaining code files implement the global-rewards-based hybrid multi-agent Soft Actor-Critic algorithm with credit assignment based on COMA, which can be executed using `main.py` with arguments as the exemplary ones in `args_RL_XX_small/large_zones.txt` (see comments in `main.py` for explanations of the arguments). Large parts of the code are based on code from this [GitHub repository](https://github.com/tumBAIS/HybridMADRL-AMoD), `trainer.py` and `sac_discrete.py` are partly based on code from this [GitHub repository](https://github.com/keiohta/tf2rl)

The directory `data` contains pre-processed data for the problem instances considered in the paper.

## Installation Instructions
Executing the code requires Python and the Python packages in `requirements.txt`, which can be installed with `pip install -r requirements.txt`. 
These packages include TensorFlow. In case of problems when trying to install TensorFlow, please refer to this [help page](https://www.tensorflow.org/install/errors).

## Code Execution
To run the code with arguments `args.txt`, execute `python main.py @args.txt` in the `algorithms` directory (analogously for the greedy algorithm). 

For typical instance and neural network sizes, a GPU should be used.
