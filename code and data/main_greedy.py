import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--episode_length', type=int) # episode length in seconds
parser.add_argument('--time_step_size', type=int) # time step size in seconds
parser.add_argument('--veh_count', type=int) # no. of vehicles as integer
parser.add_argument('--downscaling_factor', type=int) # downscaling factor for trip data (not needed for algorithm, only included to keep track of it)
parser.add_argument('--max_req_count', type=int) # max. no. of requests per time step
parser.add_argument('--max_waiting_time', type=int) # max. waiting time in seconds
parser.add_argument('--cost_parameter', type=float) # mileage-dependent cost for maintenance etc. in USD per meter
parser.add_argument('--data_dir', type=str) # relative path to directory where data is stored
parser.add_argument('--results_dir', type=str) # relative path to directory where results shall be saved


args, unknown = parser.parse_known_args()
args = vars(args)

vehicles_list = [args["veh_count"]]

from environment import Environment
from trainer_greedy import Trainer

env = Environment(args)
trainer = Trainer(env, args)

trainer()