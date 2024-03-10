"""Parse parameters and run algorithm"""

# parse parameters
import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--random_seed', type=int) # Random seed
parser.add_argument('--episode_length', type=int) # Episode length in seconds
parser.add_argument('--time_step_size', type=int) # Time step size in seconds
parser.add_argument('--veh_count', type=int) # No. of vehicles
parser.add_argument('--downscaling_factor', type=int) # Downscaling factor for trip data (not needed for algorithm, only included to keep track of it)
parser.add_argument('--max_req_count', type=int) # Max. no. of requests per time step
parser.add_argument('--max_waiting_time', type=int) # Max. waiting time in seconds
parser.add_argument('--cost_parameter', type=float) # Mileage-dependent cost for maintenance etc. in USD per meter
parser.add_argument('--max_steps', type=int) # No. of steps to interact with environment
parser.add_argument('--min_steps', type=int) # No. of steps before neural net weight updates begin
parser.add_argument('--random_steps', type=int) # No. of steps with random policy at the beginning
parser.add_argument('--update_interval', type=int) # No. of steps between neural net weight updates
parser.add_argument('--validation_interval', type=int) # No. of steps between validation runs (must be multiple of no. of time steps per episode)
parser.add_argument('--tracking_interval', type=int) # Interval at which training data is saved 
parser.add_argument('--profile_interval', type=int) # Interval at which training step is profiled
parser.add_argument('--post_processing', type=str, choices=['simple_matching', 'weighted_matching']) # Whether final decisions are obtained through maximum bipartite matching or weighted bipartite matching
parser.add_argument('--attention', type=str) # Whether attention layer is used
parser.add_argument('--req_embedding_dim', type=int) # Units of request embedding layer
parser.add_argument('--veh_embedding_dim', type=int) # Units of vehicle embedding layer
parser.add_argument('--req_context_dim', type=int) # First dim of W in request context layer
parser.add_argument('--veh_context_dim', type=int) # First dim of W in vehicle context layer
parser.add_argument('--inner_units', type=str) # Units of inner network (sequence of feedforward layers)
parser.add_argument('--regularization_coefficient', type=float) # Coefficient for L2 regularization of networks (0 if no regularization)
parser.add_argument('--rb_size', type=int) # Replay buffer size
parser.add_argument('--batch_size', type=int) # (Mini-)batch size
parser.add_argument('--log_alpha', type=float) # log(alpha)
parser.add_argument('--tau', type=float) # Smoothing factor for exponential moving average to update target critic parameters
parser.add_argument('--huber_delta', type=float) # Delta value at which Huber loss becomes linear
parser.add_argument('--gradient_clipping', type=str) # Whether gradient clipping is applied
parser.add_argument('--clip_norm', type=float) # Global norm used for gradient clipping
parser.add_argument('--lr_a', type=float) # Learning rate for actor
parser.add_argument('--lr_c', type=float) # Learning rate for critic
parser.add_argument('--scheduled_discount', type=str) # Whether discount factor follows a schedule
parser.add_argument('--discount', type=float) # Discount factor (if scheduled, this is the start value)
parser.add_argument('--scheduled_discount_values', type=str) # List of discount factor values to be set at chosen steps (if scheduled)
parser.add_argument('--scheduled_discount_steps', type=str) # List of steps at which discount factor is set to chosen value (if scheduled)
parser.add_argument('--normalized_rews', type=str) # Whether rewards are normalized when sampled from replay buffer (if so, they are divided by the standard deviation of rewards currently stored in the replay buffer)
parser.add_argument('--data_dir', type=str) # Relative path to directory where data is stored
parser.add_argument('--results_dir', type=str) # Relative path to directory where results shall be saved
parser.add_argument('--model_dir', type=str, default=None) # Relative path to directory with saved model that shall be restored in the beginning (overwriting default initialization of network weights)
parser.add_argument('--algorithm_type', type=str, choices=["LRA", "GRA", "LRGA", "COMA^nve", "COMA^equ", "COMA^tgt", "COMA^adj", "COMA^scd"]) # Specify used algorithm: LRA: local rewards, GRA: global rewards without baseline, LRGA: static local-global mix,
# COMA^nve: naive COMA baseline (no convergence), COMA^equ: COMA with equally-weighted baseline, COMA^tgt: COMA with target network baseline, COMA^adj: COMA adjusted (mix of COMA^equ and COMA^tgt), COMA^scd: scheduled COMA (dynamic local-COMA mix)
parser.add_argument('--avg_rew', type=str, default="True") # Spefify use of rewards divided by average number of non-zero rewards per observation in the replay buffer as global rewards
parser.add_argument('--struct_analysis', type=str, default="False") # Specify whether tests of the structural analysis should be conducted
parser.add_argument('--share_glob_rew', type=float) # Global reward share for LRGA
parser.add_argument('--xi', type=float) # Smoothing factor for exponential moving average to update target actor parameters (for COMA^tgt, COMA^adj, COMA^scd)
parser.add_argument('--beta_param', type=float) # Beta parameter for COMA^adj and COMA^scd (determines shape of weights update)
parser.add_argument('--kappa_exponent', type=float) # Kappa parameter for COMA^scd (determines shape of weights update)
parser.add_argument('--kappa_jump', type=float) # Parameter for jumps of COMA^scd

args = vars(parser.parse_args())

args["inner_units"] = [int(i) for i in args["inner_units"].split(',')]
args["scheduled_discount_values"] = [float(i) for i in args["scheduled_discount_values"].split(',')]
args["scheduled_discount_steps"] = [int(i) for i in args["scheduled_discount_steps"].split(',')]

if args["attention"] == "False":
    args["attention"] = False
elif args["attention"] == "True":
    args["attention"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --attention.')

if args["gradient_clipping"] == "False":
    args["gradient_clipping"] = False
elif args["gradient_clipping"] == "True":
    args["gradient_clipping"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --gradient_clipping.')

if args["scheduled_discount"] == "False":
    args["scheduled_discount"] = False
elif args["scheduled_discount"] == "True":
    args["scheduled_discount"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --scheduled_discount.')

if args["normalized_rews"] == "False":
    args["normalized_rews"] = False
elif args["normalized_rews"] == "True":
    args["normalized_rews"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --normalized_rews.')

if args["avg_rew"] == "False":
    args["avg_rew"] = False
elif args["avg_rew"] == "True":
    args["avg_rew"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --avg_rew.')

if args["struct_analysis"] == "False":
    args["struct_analysis"] = False
elif args["struct_analysis"] == "True":
    args["struct_analysis"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --struct_analysis.')

# set seed and further global settings
seed = args["random_seed"]

import os
os.environ['PYTHONHASHSEED'] = str(seed)

os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2" # enable XLA

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

tf.keras.mixed_precision.set_global_policy('mixed_float16') # enable mixed precision computations

# initialize environment, Soft Actor-Critic and trainer classes
from environment import Environment
from sac_discrete import SACDiscrete
from trainer import Trainer

env = Environment(args)
policy = SACDiscrete(args, env)
trainer = Trainer(policy, env, args)

# call trainer
trainer()