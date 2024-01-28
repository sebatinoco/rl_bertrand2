import pandas as pd
import yaml
from utils.get_comparison import get_rolling
import matplotlib.pyplot as plt

def plot_deviate(figsize = (6,4)):

    with open(f"configs/bertrand_dqn_deviate.yaml", 'r') as file:
        args = yaml.safe_load(file)
        deviate_start = args['train']['deviate_start']

    df = pd.read_csv('metrics/bertrand_dqn_deviate_1.csv', sep = ';')

    prices_0 = df['prices_0']
    prices_1 = df['prices_1']
    nash = df['p_nash']
    monopoly = df['p_monopoly']
    delta = df['delta']

    deviate_step = int(deviate_start * df.shape[0])

    prices_0 = prices_0[deviate_step - 30:deviate_step + 30] 
    prices_1 = prices_1[deviate_step - 30:deviate_step + 30] 
    nash = nash[deviate_step - 30:deviate_step + 30]
    monopoly = monopoly[deviate_step - 30:deviate_step + 30]
    delta = delta[deviate_step - 30:deviate_step + 30]
    x_range = range(deviate_step - 30, deviate_step + 30)

    plt.figure(figsize = figsize)
    plt.plot(x_range, prices_0, label = 'Agent 0')
    plt.plot(x_range, prices_1, label = 'Agent 1')
    plt.plot(x_range, nash, label = 'Nash')
    plt.plot(x_range, monopoly, label = 'Monopoly')
    plt.axvline(deviate_step, color = 'purple', linestyle = '--', label = 'Agent 0 Deviation')
    plt.xlabel('Timesteps')
    plt.ylabel('Prices')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.savefig(f'figures/simple_experiments/deviate_prices.pdf')
    plt.close()
    
    plt.figure(figsize = figsize)
    plt.plot(x_range, delta, label = 'Average profits')
    plt.axhline(1, color = 'red', label = 'Monopoly')
    plt.axhline(0, color = 'green', label = 'Nash')
    plt.axvline(deviate_step, color = 'purple', linestyle = '--', label = 'Agent 0 Deviation')
    plt.xlabel('Timesteps')
    plt.ylabel('Prices')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.savefig(f'figures/simple_experiments/deviate_delta.pdf')
    plt.close()