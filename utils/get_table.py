import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from utils.get_plots import get_rolling, get_rolling_std

def get_tables(envs = None, models = None, window_size = 1000, percent = 0.1):

    if envs is None:
        envs = list(set([metric.split('_')[0] for metric in os.listdir('metrics/single') if '.gitignore' not in metric]))
    if models is None:
        models = list(set([metric.split('_')[1] for metric in os.listdir('metrics/single') if '.gitignore' not in metric]))

    for env in envs:
        for model in models:
            metrics = os.listdir('metrics/single')
            #metrics = [metric for metric in metrics if (env in metric and model in metric) and ('altruist' not in metric) and ('deviate' not in metric)]
            metrics = [metric for metric in metrics if (env in metric and model in metric)]

            delta_base = pd.read_csv(f'metrics/single/{env}_{model}_base_1.csv', sep = ';')['delta']
            delta_base = delta_base.tail(int(delta_base.shape[0] * percent))

            results = {}
            for metric in metrics:
                df_metric = pd.read_csv(f'metrics/single/{metric}', sep = ';')
                price_cols = [col for col in df_metric.columns if 'prices' in col]
                df_metric['price_avg'] = get_rolling(df_metric[price_cols].mean(axis = 1), window_size)            
                
                N_min = np.argmax(get_rolling(df_metric['delta'], 1000) > 0)
                
                tail_percent = int(df_metric.shape[0] * percent)
                
                df_tail = df_metric.tail(tail_percent) if percent < 1 else df_metric
                
                avg_delta = np.round(np.mean(df_tail['delta']), 2)
                std_delta = np.round(np.std(df_tail['delta']), 2)
                P_avg = np.round(np.mean(df_tail['price_avg']), 2)
                
                t_statistic, p_value = stats.ttest_ind(delta_base, df_tail['delta'])
                p_value = np.round(p_value, 2)
                
                results[metric] = [avg_delta, p_value, std_delta, P_avg, N_min]
                
            df_results = pd.DataFrame(results.values(), index = results.keys(), columns=['Delta', 'p_value', 'Std', 'P_avg', 'N_min'])
            df_results = df_results.sort_values(by = 'Delta', ascending = False)

            df_results.to_csv(f'tables/{env}_{model}_results.txt', sep = '|')