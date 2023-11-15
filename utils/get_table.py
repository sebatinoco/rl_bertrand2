import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def get_table():

    env = 'linear'
    metrics = os.listdir('metrics')
    metrics = [metric for metric in metrics if (env in metric) and ('altruist' not in metric) and ('deviate' not in metric)]

    delta_base = pd.read_csv('metrics/linear_sac_base_1.csv', sep = ';')['delta']

    results = {}
    for metric in metrics:
        
        df_metric = pd.read_csv(f'metrics/{metric}', sep = ';')
        delta_metric = df_metric['delta']
        avg_delta = np.round(np.mean(delta_metric), 2)
        
        t_statistic, p_value = stats.ttest_ind(delta_base, delta_metric)
        p_value = np.round(p_value, 2)
        
        results[metric] = [avg_delta, p_value]
        
    df_results = pd.DataFrame(results.values(), index = results.keys(), columns=['Delta', 'p_value'])
    df_results = df_results.sort_values(by = 'Delta', ascending = False)

    df_results.to_csv('hyperparams_results.txt', sep = '|')