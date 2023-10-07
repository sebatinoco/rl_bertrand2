import os
import yaml
import torch
import time

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

models_dict = {'sac': SACAgent, 'ddpg': DDPGAgent, 'dqn': DQNAgent}
envs_dict = {'bertrand': BertrandEnv, 'linear': LinearBertrandEnv}

if __name__ == '__main__':
    
    configs = sorted(os.listdir('configs'))
    
    r_args = run_args() 
    
    train_agents = r_args['train_agents']
    filter_env = r_args['env']
    filter_config = r_args['filter_config']
    nb_experiments = r_args['nb_experiments']
    window_size = r_args['window_size']
    metrics_folder = r_args['metrics_folder']
    device = f"cuda:{r_args['gpu']}" if torch.cuda.is_available() else 'cpu'
    print(f'using {device}!')
    
    if train_agents:
        # filter configs if specified
        if filter_env or filter_config:
            env_configs = [config for config in configs if len(set(filter_env) & set(config.split('_'))) > 0] # filter by environment
            filtered_configs = [config for config in configs if config in filter_config] # filter by config
            
            final_configs = set(env_configs + filtered_configs) # filtered configs
            configs = [config for config in configs if config in final_configs] # filter configs

        print('Running experiments on the following configs: ', configs)
        
        failed_experiments = []
        for experiment_idx in range(1, nb_experiments + 1):
            start_time = time.time()
            # load config
            for config in configs:
                #for trigger_deviation in [True, False, 'altruist']:
                for trigger_deviation in ['altruist']:
                    trigger_name = ''
                    if trigger_deviation == True:
                        trigger_name = '_deviate'
                    elif trigger_deviation == 'altruist':
                        trigger_name = '_altruist'
                    
                    with open(f"configs/{config}", 'r') as file:
                        args = yaml.safe_load(file)
                        agent_args = args['agent']
                        env_args = args['env']
                        buffer_args = args['buffer']
                        train_args = args['train']

                    #train_args['timesteps'] = 500
                    #train_args['episodes'] = 1

                    # set experiment name
                    exp_name = f"{args['env_name']}_{args['exp_name']}_{experiment_idx}" + trigger_name
                    
                    # load model
                    model = models_dict[args['model']] 
                    
                    # load environment, agent and buffer
                    env = envs_dict[args['env_name']]
                    #env = envs_dict['bertrand']
                    env = env(**env_args, timesteps = train_args['timesteps'])      

                    dim_states = (env.N * env.k) + env.k + 1
                    dim_actions = args['n_actions'] if args['model'] == 'dqn' else 1
                    
                    agents = [model(dim_states, dim_actions, **agent_args) for _ in range(env.N)]
                    buffer = ReplayBuffer(dim_states = dim_states, N = env.N, **buffer_args)
                    
                    # train
                    try:
                        train(env, agents, buffer, env.N, exp_name = exp_name, trigger_deviation = trigger_deviation, **train_args)
                    except:
                        failed_experiments.append(exp_name)
                
            execution_time = time.time() - start_time

            print('\n' + f'{execution_time:.2f} seconds -- {(execution_time/60):.2f} minutes -- {(execution_time/3600):.2f} hours') 
            print('Failed experiments:', failed_experiments) 
    
    # filter metrics data
    metrics = [metric.replace('.csv', '') for metric in os.listdir('metrics') if ('.csv' in metric) & ('experiment' not in metric)]
    
    # plot
    for metric in metrics:
        get_plots(metric, window_size = window_size, metrics_folder=metrics_folder)
        
    get_comparison(window_size = window_size, metrics_folder=metrics_folder)
        
    folder_size_mb = get_folder_size('./metrics')
    print(f"Metrics folder size: {folder_size_mb:.2f} MB")