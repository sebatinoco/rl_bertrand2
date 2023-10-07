import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.plot_metrics import get_rolling

dict_labels = {
    'lr': {
        'base': 'lr = 0.0001',
        'lr_1': 'lr = 0.001',
        'lr_2': 'lr = 0.005'
        },
    'N': {
        'base': 'N = 2',
        'N_1': 'N = 5',
        'N_2': 'N = 10'  
    },
    'gamma': {
        'base': '$\gamma$ = 0.99',
        'gamma_1': '$\gamma$ = 0.9',
        'gamma_2': '$\gamma$ = 0.8'
    },
    'rho': {
        'base': 'rho = 0.0001',
        'rho_1': 'rho = 0.0002',
        'rho_2': 'rho = 0.0005'
        },
    'k': {
        'base': 'k = 1',
        'k_1': 'k = 5',
        'k_2': 'k = 10'
    },
}

def get_avg(config, n_experiments):
    
    '''
    Returns the average metric per step between all experiments
    '''
    
    files = [config + f'_{idx}' for idx in range(1, n_experiments + 1)]

    dataframe = pd.DataFrame()
    for file in files:
        df_tmp = pd.read_csv(f'metrics/{file}.csv', sep = '\t')
        dataframe[f'metric_{file}'] = df_tmp['metric_history']

    return dataframe.mean(axis = 1)

def plot_delta(model, param, n_experiments):
    exp_name = f'{model}_{param}'
    configs = sorted([config.replace('.yaml', '') for config in os.listdir('configs') if (exp_name in config) & ('trigger' not in config)])
    configs = [f'{model}_base'] + configs

    plt.figure(figsize = (12, 4))

    for config in configs:
        config_series = get_rolling(get_avg(config, n_experiments), 1000)
        config_key = config.replace(f'{model}_', '') # key to label dict
        plt.plot(config_series, 
                 label = dict_labels[param][config_key]
                 ) 
        
    variable = exp_name.split('_')[1]

    plt.title(f'Collusion Grade for different levels of {variable} (Rolling window of 1000)')
    plt.xlabel('Timesteps')
    plt.ylabel('Profit Gain $\Delta$')
    plt.legend()
    plt.savefig(f'figures/agg_experiments/{exp_name}_metric.pdf')
    
def get_avg_table(n_experiments):
    models = set([cfg.split('_')[0] for cfg in os.listdir('configs')])

    for model in models:
        configs = [config.replace('.yaml', '') for config in os.listdir('configs') if ('trigger' not in config) & (model in config)]
        avg_dict = {}
        for config in configs:
            files = [config + f'_{idx}' for idx in range(1, n_experiments + 1)]
            averages = []
            for file in files:
                df_tmp = pd.read_csv(f'metrics/{file}.csv', sep = '\t')
                metric_array = df_tmp['metric_history']
                avg_metric = np.mean(metric_array)
                
                averages += [avg_metric]
                
            avg_dict[config] = np.mean(averages)

        results = pd.DataFrame(avg_dict, index = ['metric']).T
        results['base_diff'] = results['metric'] - avg_dict['sac_base']
        results.sort_values(by = 'base_diff', ascending = False)
        
        results.to_excel(f'metrics/tables/{model}.xlsx', index = False)
        
def plot_results(n_experiments):
    
    models = set([config.split('_')[0] for config in os.listdir('configs')])
    params = set([config.split('_')[1] for config in os.listdir('configs') if 'base' not in config])
        
    for model in models:
        for param in params:
            plot_delta(model, param, n_experiments)
    
    get_avg_table(n_experiments)