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

    envs = ['bertrand', 'linear']
    parameters = ['N', 'gamma', 'rho', 'lr', 'k']
    for env in envs:
        env_metrics = [metric for metric in metrics if env in metric]
        for parameter in parameters:
            parameter_metrics = sorted([metric for metric in env_metrics if parameter in metric or 'base' in metric])
            for deviate in [True, False, 'altruist']:
                trigger_name = ''
                if deviate == True:
                    trigger_name = '_deviate'
                    final_metrics = sorted([metric for metric in parameter_metrics if 'deviate' in metric])
                elif deviate == 'altruist':
                    trigger_name = '_altruist'
                    final_metrics = sorted([metric for metric in parameter_metrics if 'altruist' in metric])
                else:
                    final_metrics = sorted([metric for metric in parameter_metrics if 
                                            (('deviate' not in metric) and ('altruist' not in metric)) or 
                                        ('base' in metric and 'deviate' not in metric and 'altruist' not in metric)])

                plt.figure(figsize = (12, 4))
                for file in final_metrics:
                    delta_serie = pd.read_csv(f'{metrics_folder}/' + file, sep = ';')['delta']#[-10000:]
                    #delta_serie = get_rolling(delta_serie, window_size)
                    delta_avg = get_rolling(delta_serie, window_size)
                    delta_std = get_rolling_std(delta_serie, window_size)
                    series_size = len(delta_avg)
                    plt.errorbar(range(series_size), delta_avg, delta_std, errorevery=int(0.01 * series_size), label = get_label(file, parameter))
                    #plt.plot(delta_serie, label = get_label(file, parameter))
                plt.plot([1 for i in range(delta_serie.shape[0])], label = 'Monopoly profits', color = 'red')
                plt.plot([0 for i in range(delta_serie.shape[0])], label = 'Nash profits', color = 'green')
                #plt.axhline(1, label = 'Monopoly profits', color = 'red')
                #plt.axhline(0, label = 'Nash profits', color = 'green')
                plt.xlabel('Timesteps')
                plt.ylabel('Delta')
                plt.legend()
                plt.savefig(f'figures/agg_experiments/{env}_{parameter}' + trigger_name + '.pdf')
                plt.close()