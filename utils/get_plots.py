import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_rolling(series, window_size):
  
  '''
  Returns the rolling average of actions using a fixed window_size.
  '''

  rolling_avg = np.convolve(series, np.ones(window_size)/window_size, mode = 'valid')

  fill = np.full([window_size - 1], np.nan)
  rolling_avg = np.concatenate((fill, rolling_avg))

  return rolling_avg

def get_rolling_std(series, window_size):
    '''
    Returns the rolling standard deviation using a fixed window_size.
    '''
    rolling_mean = get_rolling(series, window_size)
    squared_diff = (series - rolling_mean) ** 2
    rolling_variance = get_rolling(squared_diff, window_size)
    rolling_std = np.sqrt(rolling_variance)

    return rolling_std

def get_plots(exp_name, window_size = 500, metrics_folder = 'metrics', figsize = (8, 4)):

    ###########################################
    exp_name = exp_name.replace('.csv','')
    path = f'{metrics_folder}/{exp_name}.csv'
    df_plot = pd.read_csv(f'{path}', sep = ';', encoding = 'utf-8-sig')
    df_avg = pd.DataFrame()
    df_std = pd.DataFrame()
    
    actions_cols = [col for col in df_plot.columns if 'actions' in col]
    price_cols = [col for col in df_plot.columns if 'prices' in col]
    rewards_cols = [col for col in df_plot.columns if 'rewards' in col]
    quantities_cols = [col for col in df_plot.columns if 'quantities' in col]

    n_agents = len(actions_cols)

    df_plot['avg_actions'] = df_plot[actions_cols].mean(axis = 1)
    df_plot['avg_prices'] = df_plot[price_cols].mean(axis = 1)
    df_plot['avg_rewards'] = df_plot[rewards_cols].mean(axis = 1)
    df_plot['avg_quantities'] = df_plot[quantities_cols].mean(axis = 1)
    avg_cols = [col for col in df_plot.columns if 'avg' in col]

    window_cols = price_cols + rewards_cols + quantities_cols + avg_cols + ['delta']
    for col in window_cols:
        df_avg[col] = get_rolling(df_plot[col], window_size = window_size)
        df_std[col] = get_rolling_std(df_plot[col], window_size = window_size)
        
    series_size = df_avg.shape[0]
        
    ############################################
    plt.figure(figsize = figsize)
    for agent in range(n_agents):
        serie = f'prices_{agent}'
        plt.errorbar(range(series_size), df_avg[serie], df_std[serie], errorevery=int(0.01 * series_size), label = f'Agent {agent}')
    plt.plot(df_plot['p_monopoly'], color = 'red', label = 'Monopoly')
    plt.plot(df_plot['p_nash'], color = 'green', label = 'Nash')
    plt.xlabel('Timesteps')
    plt.ylabel('Prices')
    plt.legend(loc = 'lower right')
    plt.savefig(f'figures/simple_experiments/{exp_name}_prices.pdf')
    plt.close()
    
    ############################################
    plt.figure(figsize = figsize)
    plt.errorbar(range(series_size), df_avg['avg_prices'], df_std['avg_prices'], errorevery=int(0.01 * series_size), label = f'Average prices')
    plt.plot(df_plot['p_monopoly'], color = 'red', label = 'Monopoly')
    plt.plot(df_plot['p_nash'], color = 'green', label = 'Nash')
    plt.xlabel('Timesteps')
    plt.ylabel('Prices')
    plt.legend(loc = 'lower right')
    plt.savefig(f'figures/simple_experiments/{exp_name}_avg_prices.pdf')
    plt.close()
    
    ############################################
    plt.figure(figsize = figsize)
    plt.errorbar(range(series_size), df_avg['avg_rewards'], df_std['avg_rewards'], errorevery=int(0.01 * series_size), label = f'Average profits')
    plt.plot(df_plot['pi_N'], label = 'Nash', color = 'green')
    plt.plot(df_plot['pi_M'], label = 'Monopoly', color = 'red')
    plt.xlabel('Timesteps')
    plt.ylabel('Profits')
    plt.legend(loc = 'lower right')
    plt.savefig(f'figures/simple_experiments/{exp_name}_rewards.pdf')
    plt.close()
    
    ############################################
    plt.figure(figsize = figsize)
    plt.errorbar(range(series_size), df_avg['delta'], df_std['delta'], errorevery=int(0.01 * series_size), label = f'Average profits')
    plt.axhline(1, color = 'red', label = 'Nash')
    plt.axhline(0, color = 'green', label = 'Monoply')
    plt.xlabel('Timesteps')
    plt.ylabel('Delta')
    plt.legend(loc = 'lower right')
    plt.savefig(f'figures/simple_experiments/{exp_name}_delta.pdf')
    plt.close()