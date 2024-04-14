import os
import yaml
import torch
import time
from tqdm import tqdm
import numpy as np
import shutil

from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.dqn import DQNAgent

#from envs.BertrandInflation import BertrandEnv
from envs.BertrandInflation2 import BertrandEnv
from envs.LinearBertrandInflation import LinearBertrandEnv

from replay_buffer import ReplayBuffer
from utils.run_args import run_args
from utils.train import train
from utils.get_plots import get_plots
from utils.get_folder_size import get_folder_size
from utils.get_comparison import get_comparison
from utils.get_table import get_tables
from utils.train_test_seed_agents import train_test_seed_agents, plot_train_test
from utils.plot_deviate import plot_deviate
from utils.robust_utils import *
from utils.get_test_results import *

models_dict = {'sac': SACAgent, 'ddpg': DDPGAgent, 'dqn': DQNAgent}
envs_dict = {'bertrand': BertrandEnv, 'linear': LinearBertrandEnv}

if __name__ == '__main__':
    
    configs = sorted(os.listdir('configs'))

    r_args = run_args() 
    
    train_agents = r_args['train_agents']
    filter_env = r_args['env']
    filter_model = r_args['model']
    filter_config = r_args['filter_config']
    nb_experiments = r_args['nb_experiments']
    window_size = r_args['window_size']
    metrics_folder = r_args['metrics_folder']
    random_state = r_args['random_state']
    get_test = r_args['get_test']
    debug = r_args['debug']
    export = r_args['export']
    n_pairs = r_args['n_pairs']
    n_intervals = r_args['n_intervals']
    test_timesteps = r_args['test_timesteps']
    device = f"cuda:{r_args['gpu']}" if torch.cuda.is_available() else 'cpu'
    print(f'using {device}!')
    
    if filter_env or filter_model or filter_config:
        env_configs = [config for config in configs if len(set(filter_env) & set(config.split('_'))) > 0] or configs # filter by environment
        model_configs = [config for config in configs if len(set(filter_model) & set(config.split('_'))) > 0] or configs # filter by env
        filtered_configs = [config for config in configs if config in filter_config] or configs # filter by config
        
        final_configs = set(env_configs) & set(model_configs) & set(filtered_configs) # filtered configs
        configs = [config for config in configs if config in final_configs] # filter configs
    
    #if 'bertrand_dqn_deviate.yaml' in configs:
    #    configs.remove('bertrand_dqn_deviate.yaml')

    print('Running experiments on the following configs: ', configs)

    if train_agents:
        # base seed
        np.random.seed(random_state)
        seeds = np.random.randint(0, 100_000, nb_experiments) # 1 seed for each n° experiment
        for config in configs:
            # load config
            with open(f"configs/{config}", 'r') as file:
                args = yaml.safe_load(file)
                agent_args = args['agent']
                env_args = args['env']
                buffer_args = args['buffer']
                train_args = args['train']
                variation = args['variation']
                random_state = args['random_state']

            for experiment_idx in range(1, nb_experiments + 1):

                train_args['timesteps'] = 2000

                # random seed
                random_seed = int(seeds[experiment_idx - 1])

                # set experiment name
                exp_name = f"{args['env_name']}_{args['exp_name']}_{experiment_idx}"
                
                # load model
                model = models_dict[args['model']] 
                
                # dimensions
                dim_states = (env_args['N'] * env_args['k']) + env_args['k'] + 1
                dim_actions = args['n_actions'] if args['model'] == 'dqn' else 1
                
                # load environment, agent and buffer
                env = envs_dict[args['env_name']]
                env = env(**env_args, timesteps = train_args['timesteps'], dim_actions = dim_actions, random_state = random_seed)      
                
                agents = [model(dim_states, dim_actions, **agent_args, random_state = random_seed + _, device = device) for _ in range(env.N)]
                buffer = ReplayBuffer(dim_states = dim_states, N = env.N, **buffer_args)
                
                # train
                train(env, agents, buffer, env.N, exp_name = exp_name, variation = variation, debug=debug, robust=True, **train_args)

    if get_test:
        # create test metrics as .parquet
        get_test_results(random_state = random_state, n_intervals = n_intervals, 
                         n_pairs = n_pairs, test_timesteps = test_timesteps, 
                         configs = configs, device = device)

    print('getting results...')
     # get test curve
    get_test_metrics(test_timesteps = test_timesteps, n_pairs = n_pairs,
                     n_intervals = n_intervals, n_bins = 1)
    
    get_test_tables(nb_experiments = n_pairs)
    
    get_robust_plots(window_size = window_size)
    get_train_tables(nb_experiments)


    if export:
        print('exporting files...')
        # move to export folder
        shutil.make_archive("export/metrics", "zip", "metrics")
        shutil.make_archive("export/models", "zip", "models")

    print('done!')