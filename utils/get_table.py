import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def get_tables(percent = 0.1):

    envs = list(set([metric.split('_')[0] for metric in os.listdir('metrics') if '.gitignore' not in metric]))
    models = list(set([metric.split('_')[1] for metric in os.listdir('metrics') if '.gitignore' not in metric]))

    for env in envs:
        for model in models:
            metrics = os.listdir('metrics')
            #metrics = [metric for metric in metrics if (env in metric and model in metric) and ('altruist' not in metric) and ('deviate' not in metric)]
            metrics = [metric for metric in metrics if (env in metric and model in metric)]

            delta_base = pd.read_csv(f'metrics/{env}_{model}_base_1.csv', sep = ';')['delta']

            results = {}
            for metric in metrics:
                df_metric = pd.read_csv(f'metrics/{metric}', sep = ';')
                tail_percent = int(df_metric.shape[0] * percent)
                delta_metric = df_metric['delta'].tail(tail_percent) if percent < 1 else df_metric['delta']
                avg_delta = np.round(np.mean(delta_metric), 2)
                
                t_statistic, p_value = stats.ttest_ind(delta_base, delta_metric)
                p_value = np.round(p_value, 2)
                
                results[metric] = [avg_delta, p_value]
                
            df_results = pd.DataFrame(results.values(), index = results.keys(), columns=['Delta', 'p_value'])
            df_results = df_results.sort_values(by = 'Delta', ascending = False)

            df_results.to_csv(f'tables/{env}_{model}_results.txt', sep = '|')