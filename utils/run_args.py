import argparse
from distutils.util import strtobool

def run_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--env', nargs = '+', default = [], help = 'environments to run the experiments')
    parser.add_argument('--model', nargs = '+', default = [], help = 'models to run the experiments')
    parser.add_argument('--filter_config', nargs = '+', default = [], help = 'specific config to run the experiments')
    parser.add_argument('--gpu', default = 0, type = int, help = 'gpu to run the experiments')
    parser.add_argument('--nb_experiments', default = 50, type = int, help = 'number of experiments per config')
    parser.add_argument('--train_agents', default = True, type=lambda x: bool(strtobool(x)), help = 'bool to execute training if wanted')
    parser.add_argument('--window_size', default = 1000, type = int, help = 'window size to plot')
    parser.add_argument('--metrics_folder', default = 'metrics/single', type = str, help = 'metrics folder')
    parser.add_argument('--random_state', default = 3381, type = int, help = 'seed for experiment')
    parser.add_argument('--debug', default = False, type=lambda x: bool(strtobool(x)), help = 'bool to debug training if wanted')
    parser.add_argument('--get_test', default = True, type=lambda x: bool(strtobool(x)), help = 'bool to get test results if wanted')
    parser.add_argument('--export', default = False, type=lambda x: bool(strtobool(x)), help = 'bool to export results if wanted')
    parser.add_argument('--n_pairs', default = 30, type = int, help = 'number of experiments for each test timestep')
    parser.add_argument('--n_intervals', default = 10, type = int, help = 'number of timesteps to test')
    parser.add_argument('--test_timesteps', default = 400000, type = int, help = 'number of test timesteps')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args