import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from utils.get_plots import get_rolling, get_rolling_std

def get_robust_table(percent = 0.05):
    
    metrics = os.listdir('metrics/robust/')
    configs = ['bertrand_dqn_base', 'bertrand_dqn_default']

    robust_results = {}
    for config in configs:
        files = [metric for metric in metrics if config in metric]
        deltas = []
        for file in files:
            result = pd.read_csv(f'metrics/robust/{file}', sep = ';')

            tail_percent = int(result.shape[0] * percent)
            df_tail = result.tail(tail_percent) if percent < 1 else result
            df_tail = result.iloc[350_000:400_000]

            delta = np.round(np.mean(df_tail['delta']), 2)
            deltas.append(delta)
        robust_results[config] = deltas
        
    table = {}
    for config in configs:
        statistic, p_value = ttest_rel(robust_results['bertrand_dqn_base'], robust_results[config]) # paired t-test
        table[config] = p_value
    
    return table

def get_robust_plots(window_size = 1000):

    act2prof = np.linspace(-50, 200, 15)

    metrics = os.listdir('metrics/robust/')
    configs = ['bertrand_dqn_base', 'bertrand_dqn_default'] # change for get_configs()

    avg_delta, std_delta = {}, {}
    avg_actions, std_actions = {}, {}
    for config in configs:
        avg_actions_tmp , std_actions_tmp = np.zeros([500_000, 40]), np.zeros([500_000, 40])
        avg_delta_tmp , std_delta_tmp = np.zeros([500_000, 40]), np.zeros([500_000, 40])
        files = [metric for metric in metrics if config in metric]
        for idx in range(len(files)):
            result = pd.read_csv(f'metrics/robust/{files[idx]}', sep = ';')
            avg_delta_tmp[:, idx] = get_rolling(result['delta'], window_size)
            std_delta_tmp[:, idx] = get_rolling_std(result['delta'], window_size)
            
            mean_actions = (result['actions_0'].apply(lambda x: act2prof[int(x)]) + result['actions_1'].apply(lambda x: act2prof[int(x)])) / 2
            avg_actions_tmp[:, idx] = get_rolling(mean_actions, window_size)
            std_actions_tmp[:, idx] = get_rolling_std(mean_actions, window_size)
            
        avg_actions[config] = avg_actions_tmp
        std_actions[config] = std_actions_tmp
        avg_delta[config] = avg_delta_tmp
        std_delta[config] = std_delta_tmp

    series_size = 500_000
    fig, ax = plt.subplots(1, 2, figsize = (15, 4))

    ax[0].errorbar(range(series_size), np.mean(avg_delta['bertrand_dqn_base'], axis = 1), np.std(avg_delta['bertrand_dqn_base'], axis = 1), errorevery=int(0.01 * series_size), label = 'Inflation')
    ax[0].errorbar(range(series_size), np.mean(avg_delta['bertrand_dqn_default'], axis = 1), np.std(avg_delta['bertrand_dqn_default'], axis = 1), errorevery=int(0.01 * series_size), label = 'No Inflation')
    ax[0].axhline(0, linestyle = '--', color = 'green', label = 'Monopoly')
    ax[0].axhline(1, linestyle = '--', color = 'red', label = 'Nash')
    ax[0].set(xlabel = 'Timesteps', ylabel = 'Avg Delta')
    ax[0].legend()
    
    ax[1].errorbar(range(series_size), np.mean(std_delta['bertrand_dqn_base'], axis = 1), np.std(std_delta['bertrand_dqn_base'], axis = 1), errorevery=int(0.01 * series_size), label = 'Inflation')
    ax[1].errorbar(range(series_size), np.mean(std_delta['bertrand_dqn_default'], axis = 1), np.std(std_delta['bertrand_dqn_default'], axis = 1), errorevery=int(0.01 * series_size), label = 'No Inflation')
    ax[1].set(xlabel = 'Timesteps', ylabel = 'Std Delta')
    ax[1].legend()
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(1, 2, figsize = (15, 4))
    ax[0].errorbar(range(series_size), np.mean(avg_actions['bertrand_dqn_base'], axis = 1), np.std(avg_actions['bertrand_dqn_base'], axis = 1), errorevery=int(0.01 * series_size), label = 'Inflation')
    ax[0].errorbar(range(series_size), np.mean(avg_actions['bertrand_dqn_default'], axis = 1), np.std(avg_actions['bertrand_dqn_default'], axis = 1), errorevery=int(0.01 * series_size), label = 'No Inflation')
    ax[0].axhline(act2prof[0], linestyle = '--', label = 'Lower Bound', color = 'green')
    ax[0].axhline(act2prof[-1], linestyle = '--', label = 'Upper Bound', color = 'red')
    ax[0].set(xlabel = 'Timesteps', ylabel = 'Avg Expected Profit (%)')
    ax[0].legend()

    ax[1].errorbar(range(series_size), np.mean(std_actions['bertrand_dqn_base'], axis = 1), np.std(std_actions['bertrand_dqn_base'], axis = 1), errorevery=int(0.01 * series_size), label = 'Inflation')
    ax[1].errorbar(range(series_size), np.mean(std_actions['bertrand_dqn_default'], axis = 1), np.std(std_actions['bertrand_dqn_default'], axis = 1), errorevery=int(0.01 * series_size), label = 'No Inflation')
    ax[1].set(xlabel = 'Timesteps', ylabel = 'Std Expected Profit (%)')
    ax[1].legend()    
    plt.show()
    plt.close()