import argparse
from distutils.util import strtobool

def run_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--env', nargs = '+', default = [], help = 'environments to run the experiments')
    parser.add_argument('--model', nargs = '+', default = [], help = 'models to run the experiments')
    parser.add_argument('--filter_config', nargs = '+', default = [], help = 'specific config to run the experiments')
    parser.add_argument('--gpu', default = 0, type = int, help = 'gpu to run the experiments')
    parser.add_argument('--nb_experiments', default = 1, type = int, help = 'number of experiments per config')
    parser.add_argument('--train_agents', default = True, type=lambda x: bool(strtobool(x)), help = 'bool to execute training if wanted')
    parser.add_argument('--window_size', default = 1000, type = int, help = 'window size to plot')
    parser.add_argument('--metrics_folder', default = 'metrics', type = str, help = 'metrics folder')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args