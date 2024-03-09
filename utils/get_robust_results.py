import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.plot_metrics import get_rolling

def get_robust_results(config, timesteps):

    config = config.replace('.yaml', '')
    nb_experiments = len([metric for metric in os.listdir('metrics/robust/') if config in metric])

    delta_data = np.zeros((timesteps, nb_experiments))
    for experiment in range(nb_experiments):
        data = pd.read_parquet(f'metrics/robust/{config}_{experiment + 1}.parquet')
        rolling_delta = get_rolling(data['delta'], window_size = 10)
        delta_data[:, experiment] = rolling_delta

    delta_avg = np.mean(delta_data, axis = 1)
    delta_std = np.std(delta_data, axis = 1)

    plt.figure(figsize = (6, 4))
    plt.errorbar(range(timesteps), delta_avg, delta_std, errorevery = int(0.01 * timesteps), label = 'Average $\Delta$')
    plt.axhline(1, color = 'red', label = 'Monopoly', linestyle = '--')
    plt.axhline(0, color = 'green', label = 'Nash', linestyle = '--')
    plt.ylabel('$\Delta$')
    plt.xlabel('Timesteps')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.savefig(f'figures/simple_experiments/robust_{config}.pdf')
    plt.close()