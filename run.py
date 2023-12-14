import os
import yaml
import torch
import time
from tqdm import tqdm
import numpy as np

from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.dqn import DQNAgent

from envs.BertrandInflation import BertrandEnv
from envs.LinearBertrandInflation import LinearBertrandEnv

from replay_buffer import ReplayBuffer
from utils.run_args import run_args
from utils.train import train
from utils.get_plots import get_plots
from utils.get_folder_size import get_folder_size
from utils.get_comparison import get_comparison
from utils.get_table import get_tables

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
    device = f"cuda:{r_args['gpu']}" if torch.cuda.is_available() else 'cpu'
    print(f'using {device}!')
    
    if train_agents:
        # filter configs if specified
        if filter_env or filter_model or filter_config:
            env_configs = [config for config in configs if len(set(filter_env) & set(config.split('_'))) > 0] or configs # filter by environment
            model_configs = [config for config in configs if len(set(filter_model) & set(config.split('_'))) > 0] or configs # filter by env
            filtered_configs = [config for config in configs if config in filter_config] or configs # filter by config
            
            final_configs = set(env_configs) & set(model_configs) & set(filtered_configs) # filtered configs
            configs = [config for config in configs if config in final_configs] # filter configs

        print('Running experiments on the following configs: ', configs)
        
        failed_experiments = []
        for experiment_idx in range(1, nb_experiments + 1):
            start_time = time.time()
            # load config
            for config in configs:
                with open(f"configs/{config}", 'r') as file:
                    args = yaml.safe_load(file)
                    agent_args = args['agent']
                    env_args = args['env']
                    buffer_args = args['buffer']
                    train_args = args['train']
                    variation = args['variation']
                    random_state = args['random_state']

                train_args['timesteps'] = 500
                train_args['episodes'] = 1
                
                # random seed
                np.random.seed(random_state)

                # set experiment name
                exp_name = f"{args['env_name']}_{args['exp_name']}_{variation}_{experiment_idx}"
                
                # load model
                model = models_dict[args['model']] 
                
                # dimensions
                dim_states = (env_args['N'] * env_args['k']) + env_args['k'] + 1
                dim_actions = args['n_actions'] if args['model'] == 'dqn' else 1
                
                # load environment, agent and buffer
                env = envs_dict[args['env_name']]
                env = env(**env_args, timesteps = train_args['timesteps'], dim_actions = dim_actions, random_state = random_state)      
                
                agents = [model(dim_states, dim_actions, **agent_args, random_state = random_state + _) for _ in range(env.N)]
                buffer = ReplayBuffer(dim_states = dim_states, N = env.N, **buffer_args)
                
                # train
                #try:
                #    train(env, agents, buffer, env.N, exp_name = exp_name, variation = variation, **train_args)
                #except:
                #    failed_experiments.append(exp_name)
                train(env, agents, buffer, env.N, exp_name = exp_name, variation = variation, **train_args)
                    
                
            execution_time = time.time() - start_time

            print('\n' + f'{execution_time:.2f} seconds -- {(execution_time/60):.2f} minutes -- {(execution_time/3600):.2f} hours') 
            print('Failed experiments:', failed_experiments) 
    
    # filter metrics data
    metrics = os.listdir('metrics')
    
    if filter_env or filter_model or filter_config:
        env_metrics = [metric for metric in metrics if len(set(filter_env) & set(metric.split('_'))) > 0] or metrics # filter by environment
        model_metrics = [metric for metric in metrics if len(set(filter_model) & set(metric.split('_'))) > 0] or metrics # filter by env
        filtered_metrics = [metric for metric in metrics if metric in filter_config] or metrics # filter by config
        
        final_metrics = set(env_metrics) & set(model_metrics) & set(filtered_metrics) # filtered configs
        metrics = [metric for metric in metrics if metric in final_metrics] # filter configs

    #metrics = [metric.replace('.csv', '') for metric in os.listdir('metrics') if ('.csv' in metric) & ('experiment' not in metric)]

    print('Plotting the following experiments: ', metrics)
    
    # plot
    print('generating plots!')
    for metric in tqdm(metrics):
        get_plots(metric, window_size = window_size, metrics_folder = metrics_folder)

    get_comparison(envs = filter_env, models = filter_model, window_size = window_size, metrics_folder = metrics_folder)
    
    print('creating tables!')
    get_tables(envs = filter_env, models = filter_model)
        
    folder_size_mb = get_folder_size('./metrics')
    print(f"Metrics folder size: {folder_size_mb:.2f} MB")