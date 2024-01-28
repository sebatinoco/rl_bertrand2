import pandas as pd
import os
import matplotlib.pyplot as plt
import yaml
from utils.get_plots import get_rolling, get_rolling_std

def get_label(file, parameter):
    if 'deviate' in file:
        file = file.replace('_deviate', '')    
    elif 'altruist' in file:
        file = file.replace('_altruist', '')
    base = file.split('.')[0][:-2]
    filename = f"configs/{base}.yaml" 
    with open(filename) as config:
        data = yaml.safe_load(config)
        if parameter in ['N', 'k', 'rho']:
            parameter_value = data['env'][parameter]
        elif parameter == 'gamma':
            parameter_value = data['agent'][parameter]
        elif parameter == 'lr':
            if 'sac' in filename:
                parameter_value = data['agent']['actor_lr']
            else:
                parameter_value = data['agent']['lr']
        
    label = f"{parameter} = {parameter_value}"
    
    return label

def get_comparison(envs= None, models = None, window_size = 1000, metrics_folder = 'metrics', percent = 0.1, figsize = (6, 4)):

    metrics = os.listdir(f'{metrics_folder}/')
    
    colors = {
        0: 'C0',
        1: 'C1',
        2: 'C4'
    }

    parameters = ['N', 'gamma', 'rho', 'lr', 'k']
    if envs is None:
        envs = list(set([metric.split('_')[0] for metric in metrics if '.csv' in metric]))
        envs = [env for env in envs if env in ['bertrand', 'linear']]
    if models is None:
        models = list(set([metric.split('_')[1] for metric in metrics if '.csv' in metric]))
        models = [model for model in models if env in ['dqn', 'sac', 'ddpg']]
    for env in envs:
        env_metrics = [metric for metric in metrics if env in metric]
        for model in models:
            model_metrics = [metric for metric in env_metrics if model in metric]
            base_metric = f'{env}_{model}_base_1.csv'
            for parameter in parameters:
                final_metrics = sorted([metric for metric in model_metrics if parameter in metric])
                
                if base_metric in metrics:
                    final_metrics = sorted(final_metrics + [f'{env}_{model}_base_1.csv'])
                
                plt.figure(figsize = figsize)
                count = 0
                for file in final_metrics:
                    delta_serie = pd.read_csv(f'{metrics_folder}/' + file, sep = ';')['delta']
                    delta_avg = get_rolling(delta_serie, window_size)
                    #delta_std = get_rolling_std(delta_serie, window_size)
                    #series_size = len(delta_avg)
                    #plt.errorbar(range(series_size), delta_avg, delta_std, errorevery=int(0.01 * series_size), label = get_label(file, parameter))
                    plt.plot(delta_avg, label = get_label(file, parameter), color = colors[count])
                    count += 1
                    
                plt.plot([1 for i in range(delta_serie.shape[0])], label = 'Monopoly', color = 'red')
                plt.plot([0 for i in range(delta_serie.shape[0])], label = 'Nash', color = 'green')
                #plt.axhline(1, label = 'Monopoly profits', color = 'red')
                #plt.axhline(0, label = 'Nash profits', color = 'green')
                plt.xlabel('Timesteps')
                plt.ylabel('Delta')
                plt.legend(loc = 'lower right')
                plt.tight_layout()
                plt.savefig(f'figures/agg_experiments/{env}_{model}_{parameter}_delta.pdf')
                plt.close()
                
                plt.figure(figsize = figsize)
                count = 0
                for file in final_metrics:
                    df_prices = pd.read_csv(f'{metrics_folder}/' + file, sep = ';')
                    price_cols = [col for col in df_prices.columns if 'prices' in col]
                    prices_avg = get_rolling(df_prices[price_cols].mean(axis = 1), window_size)
                    #plt.errorbar(range(series_size), delta_avg, delta_std, errorevery=int(0.01 * series_size), label = get_label(file, parameter))
                    plt.plot(prices_avg, label = get_label(file, parameter), color = colors[count])
                    count += 1
                plt.plot(df_prices['p_monopoly'], color = 'red', label = 'Monopoly')
                plt.plot(df_prices['p_nash'], color = 'green', label = 'Nash')
                #plt.axhline(1, label = 'Monopoly profits', color = 'red')
                #plt.axhline(0, label = 'Nash profits', color = 'green')
                plt.xlabel('Timesteps')
                plt.ylabel('Prices')
                plt.legend(loc = 'lower right')
                plt.tight_layout()
                plt.savefig(f'figures/agg_experiments/{env}_{model}_{parameter}_prices.pdf')
                plt.close()
                
                plt.figure(figsize = figsize)
                count = 0
                for file in final_metrics:
                    df_metric = pd.read_csv(f'{metrics_folder}/' + file, sep = ';')
                    tail_percent = int(df_metric.shape[0] * percent)
                    delta_serie = df_metric['delta'].tail(tail_percent) if percent < 1 else df_metric['delta']
                    delta_avg = get_rolling(delta_serie, window_size)
                    last_steps = range(df_metric.shape[0] - tail_percent, df_metric.shape[0])
                    plt.plot(last_steps, delta_avg, label = get_label(file, parameter), color = colors[count], linewidth = 2.0)
                    count += 1
                    
                plt.plot(last_steps, [1 for i in range(delta_serie.shape[0])], label = 'Monopoly', color = 'red')
                plt.plot(last_steps, [0 for i in range(delta_serie.shape[0])], label = 'Nash', color = 'green')
                plt.xlabel('Timesteps')
                plt.ylabel('Delta')
                plt.legend(loc = 'lower right')
                plt.tight_layout()
                plt.savefig(f'figures/agg_experiments/{env}_{model}_{parameter}_last_delta.pdf')
                plt.close()
                
                plt.figure(figsize = figsize)
                count = 0
                for file in final_metrics:
                    df_prices = pd.read_csv(f'{metrics_folder}/' + file, sep = ';')
                    tail_percent = int(df_prices.shape[0] * percent)
                    df_prices = df_prices.tail(tail_percent) if percent < 1 else df_prices['delta']
                    price_cols = [col for col in df_prices.columns if 'prices' in col]
                    prices_avg = get_rolling(df_prices[price_cols].mean(axis = 1), window_size)
                    last_steps = range(df_metric.shape[0] - tail_percent, df_metric.shape[0])
                    plt.plot(last_steps, prices_avg, label = get_label(file, parameter), color = colors[count], linewidth = 2.0)
                    count += 1
                    
                plt.plot(last_steps, df_prices['p_monopoly'], color = 'red', label = 'Monopoly')
                plt.plot(last_steps, df_prices['p_nash'], color = 'green', label = 'Nash')
                plt.xlabel('Timesteps')
                plt.ylabel('Delta')
                plt.legend(loc = 'lower right')
                plt.tight_layout()
                plt.savefig(f'figures/agg_experiments/{env}_{model}_{parameter}_last_prices.pdf')
                plt.close()