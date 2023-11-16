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
    with open(f"configs/{base}.yaml") as config:
        data = yaml.safe_load(config)
        if parameter in ['N', 'k', 'rho']:
            parameter_value = data['env'][parameter]
        elif parameter == 'gamma':
            parameter_value = data['agent'][parameter]
        elif parameter == 'lr':
            parameter_value = data['agent']['actor_lr']
        
    label = f"{parameter} = {parameter_value}"
    
    return label

def get_comparison(window_size = 1000, metrics_folder = 'metrics'):

    metrics = os.listdir(f'{metrics_folder}/')
    
    colors = {
        0: 'C0',
        1: 'C1',
        2: 'C4'
    }

    parameters = ['N', 'gamma', 'rho', 'lr', 'k']
    envs = list(set([metric.split('_')[0] for metric in metrics if '.csv' in metric]))
    models = list(set([metric.split('_')[1] for metric in metrics if '.csv' in metric]))
    for env in envs:
        env_metrics = [metric for metric in metrics if env in metric]
        for model in models:
            model_metrics = [metric for metric in env_metrics if model in metric]
            base_metric = f'{env}_{model}_base_1.csv'
            for parameter in parameters:
                final_metrics = sorted([metric for metric in model_metrics if parameter in metric])
                
                if base_metric in metrics:
                    final_metrics = sorted(final_metrics + [f'{env}_{model}_base_1.csv'])
            
            plt.figure(figsize = (8, 4))
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
            plt.legend(loc = 'upper right')
            plt.savefig(f'figures/agg_experiments/{env}_{parameter}_delta.pdf')
            plt.close()
            
            plt.figure(figsize = (8, 4))
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
            plt.savefig(f'figures/agg_experiments/{env}_{parameter}_prices.pdf')
            plt.close()