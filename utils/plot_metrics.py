import numpy as np
import matplotlib.pyplot as plt

#def get_rolling(series, window_size):
#  
#  '''
#  Returns the rolling average of actions using a fixed window_size.
#  '''
#
#  rolling_avg = np.convolve(series, np.ones(window_size)/window_size, mode = 'valid')

#  fill = np.full([window_size - 1], np.nan)
#  rolling_avg = np.concatenate((fill, rolling_avg))

#  return rolling_avg

import pandas as pd
def get_rolling(series, window_size):
  
  '''
  Returns the rolling average of actions using a fixed window_size.
  '''
  
  series = pd.Series(series)
  rolling_avg = series.rolling(window_size).mean()
  
  return rolling_avg

def get_rolling_std(series, window_size):
    '''
    Returns the rolling standard deviation using a fixed window_size.
    '''
    #rolling_mean = get_rolling(series, window_size)
    #squared_diff = (series - rolling_mean) ** 2
    #rolling_variance = get_rolling(squared_diff, window_size)
    #rolling_std = np.sqrt(rolling_variance)
    
    series = pd.Series(series)
    rolling_std = series.rolling(window_size).std()

    return rolling_std

def plot_metrics(fig, axes, prices_history, monopoly_history, nash_history, rewards_history, metric_history,
                 window_size = 1000, actor_loss = None, Q_loss = None):
  
    prices = np.array(prices_history)
    rewards = np.array(rewards_history)  
    [ax.cla() for row in axes for ax in row]
        
    for agent in range(prices.shape[1]):
      rolling_price = get_rolling(prices[:, agent], window_size)
      axes[0, 0].plot(range(rolling_price.shape[0]), rolling_price, label = f'Agent {agent}') # plot rolling avg price
      
      rolling_mean = get_rolling(rewards[:, agent], window_size)
      axes[0, 1].plot(range(rolling_mean.shape[0]), rolling_mean, label = f'Agent {agent}') # plot rolling avg reward

    axes[0, 0].plot(monopoly_history, label = 'Monopoly Price', linestyle = '--', color = 'r')
    axes[0, 0].plot(nash_history, label = 'Nash Price', linestyle = '--', color = 'g')
    axes[0, 0].set_title(f'Rolling Avg of Prices (window = {window_size})')
    axes[0, 0].set_xlabel('Timesteps')

    axes[0, 1].set_title(f'Rolling Avg of Rewards (window = {window_size})')
    axes[0, 1].set_xlabel('Timesteps')
    
    metric_mean = get_rolling(metric_history, window_size)
    axes[0, 2].plot(metric_mean, label = 'Average Profits')
    axes[0, 2].axhline(y = 1, color = 'r', linestyle = '--', label = 'Perfect Collusion')
    axes[0, 2].axhline(y = 0, color = 'g', linestyle = '--', label = 'Perfect Competition')
    axes[0, 2].set_title(f'Rolling Avg of $\Delta$ (window = {window_size})')
    axes[0, 2].set_xlabel('Timesteps')
    
    if (actor_loss is not None) & (Q_loss is not None):
      
      # graficar loss por cada agente!!

      axes[1, 0].plot(actor_loss, label = 'Actor loss')
      axes[1, 0].set_title('Actor Q(s,a) (Agent 0)')
      axes[1, 0].set_xlabel('Update iteration')

      axes[1, 1].plot(Q_loss, label = 'Critic Loss')
      axes[1, 1].set_title('Critic MSE Loss (Agent 0)')
      axes[1, 1].set_xlabel('Update iteration')
    
    [ax.grid('on') for row in axes for ax in row]
    [ax.legend(loc = 'lower right') for row in axes for ax in row]
    
    fig.tight_layout()
    plt.pause(0.05)

