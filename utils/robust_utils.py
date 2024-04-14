import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.get_plots import get_rolling, get_rolling_std
from researchpy import ttest as rpTtest

def get_configs():
    
    metrics = os.listdir('metrics/robust/')
    
    configs = ['_'.join(metric.split('_')[:4]) if 
                    ('_'.join(metric.split('_')[:3]) != 'bertrand_dqn_base') and 
                    ('_'.join(metric.split('_')[:3]) != 'bertrand_dqn_default') and
                    ('_'.join(metric.split('_')[:3]) != 'bertrand_dqn_deviate') and
                    ('_'.join(metric.split('_')[:3]) != 'bertrand_dqn_altruist') else 
                    '_'.join(metric.split('_')[:3]) for metric in metrics]
    configs.remove('.gitignore')
    configs = list(set(configs))
    
    return configs

def get_statistics(serie_1, serie_2):
    
    '''
    returns the paired t-test statistics of 2 samples
    '''
    
    serie_1 = pd.Series(serie_1)
    serie_2 = pd.Series(serie_2)
    
    results = rpTtest(serie_1, serie_2, equal_variances=True, paired=True)
    
    dz = results[1].iloc[6][1]
    d = abs(dz * (2 ** 0.5))
    g = abs(results[1].iloc[8][1])

    avg_difference = results[1].iloc[0][1]
    two_sided = results[1].iloc[3][1]
    diff_minor = results[1].iloc[4][1]
    diff_major = results[1].iloc[5][1]

    if d < .01:
        effect = 'Negligible'
    elif d < .20:
        effect = 'Very small'
    elif d < .50:
        effect = 'Small'
    elif d < .80:
        effect = 'Medium'
    elif d < 1.20:
        effect = 'Large'
    elif d < 2.00:
        effect = 'Very large'
    else:
        effect = 'Huge' 
        
    results_comparison = {
        "Average Difference": avg_difference, 
        "Two side test p value": two_sided, 
        "Difference < 0 p value": diff_minor,
        "Difference > 0 p value": diff_major,
        "Cohen's d": d, 
        'Effect': effect}
    
    results_single = dict(results[0].iloc[1, 1:])
    
    return results_single, results_comparison


def get_train_tables(nb_experiments = 50, evaluation_timesteps = 50_000):
    
    '''
    get table of robust results
    '''
    
    metrics = os.listdir('metrics/robust/')
    configs = get_configs()

    # consolidate data
    robust_delta = {} # dict with list of deltas for each config
    robust_forced_delta = {} # dict with list of forced_deltas for each config
    for config in configs:
        files = [metric.replace('.parquet', '') for metric in metrics if config in metric]
        files = sorted(files, key=lambda x: int(x.split('_')[-1])) # sort by experiment idx
        assert len(files) == nb_experiments, f'error calculating sample of experiment {config}: {len(files)} experiments' # assert of nb of experiments
        
        deltas = []
        forced_deltas = []
        for file in files:
            result = pd.read_parquet(f'metrics/robust/{file}.parquet')
            df_tail = result.iloc[-evaluation_timesteps:]

            delta = np.round(np.mean(df_tail['delta']), 2)
            forced_delta = np.round(np.mean(df_tail['forced_delta']), 2)

            deltas.append(delta)
            forced_deltas.append(forced_delta)

        robust_delta[config] = deltas
        robust_forced_delta[config] = forced_deltas
        
    # get paired t-test statistic
    delta_single_statistics, delta_comparison_statistics = {}, {}
    forced_delta_single_statistics, forced_delta_comparison_statistics = {}, {}
    for config in configs:
        delta_statistics = get_statistics(robust_delta['bertrand_dqn_base'], robust_delta[config]) # compare config against base
        forced_delta_statistics = get_statistics(robust_forced_delta['bertrand_dqn_base'], robust_forced_delta[config]) # compare config against base

        delta_single_statistics[config] = delta_statistics[0]
        delta_comparison_statistics[config] = delta_statistics[1]

        forced_delta_single_statistics[config] = forced_delta_statistics[0]
        forced_delta_comparison_statistics[config] = forced_delta_statistics[1]
        
    delta_single_statistics = pd.DataFrame(delta_single_statistics).T
    delta_comparison_statistics = pd.DataFrame(delta_comparison_statistics).T.sort_values("Cohen's d", ascending = False).dropna()

    forced_delta_single_statistics = pd.DataFrame(forced_delta_single_statistics).T
    forced_delta_comparison_statistics = pd.DataFrame(forced_delta_comparison_statistics).T.sort_values("Cohen's d", ascending = False).dropna()
    
    delta_single_statistics.to_csv('tables/train_delta_single_statistics.txt', sep = '|', encoding = 'utf-8-sig')
    delta_comparison_statistics.to_csv('tables/train_delta_comparison_statistics.txt', sep = '|', encoding = 'utf-8-sig')

    forced_delta_single_statistics.to_csv('tables/train_forced_delta_single_statistics.txt', sep = '|', encoding = 'utf-8-sig')
    forced_delta_comparison_statistics.to_csv('tables/train_forced_delta_comparison_statistics.txt', sep = '|', encoding = 'utf-8-sig')
    
def get_robust_plots(window_size = 1000, nb_experiments = 50):
    
    '''
    get plots for every config
    '''

    labels = {
        'bertrand_dqn_lr': {
            'base': '$lr = 0.001$',
            '1': '$lr = 0.002$',
            '2': '$lr = 0.003$'
        },
        'bertrand_dqn_gamma': {
            'base': '$\gamma = 0.95$',
            '1': '$\gamma = 0.80$',
            '2': '$\gamma = 0.70$'
        },
        'bertrand_dqn_k': {
            'base': '$k = 1$',
            '1': '$k = 5$',
            '2': '$k = 10$'
        },
        'bertrand_dqn_rho': {
            'base': r'$\rho = 0.001$',
            '1': r'$\rho = 0.002$',
            '2': r'$\rho = 0.003$'
        },
        'bertrand_dqn_N': {
            'base': '$N = 2$',
            '1': '$N = 3$',
            '2': '$N = 5$'
        }
    }

    act2prof = np.linspace(-50, 200, 15)

    configs = get_configs()
    metrics = [metric for metric in os.listdir('metrics/robust/') if metric.endswith('.parquet')]
    series_size = pd.read_parquet(f"metrics/robust/{metrics[0]}").shape[0]

    avg_delta, std_delta = {}, {}
    avg_forced_delta, std_forced_delta = {}, {}
    avg_actions, std_actions = {}, {}
    for config in configs:
        avg_actions_tmp , std_actions_tmp = np.zeros([series_size, nb_experiments]), np.zeros([series_size, nb_experiments])
        avg_delta_tmp , std_delta_tmp = np.zeros([series_size, nb_experiments]), np.zeros([series_size, nb_experiments])
        avg_forced_delta_tmp , std_forced_delta_tmp = np.zeros([series_size, nb_experiments]), np.zeros([series_size, nb_experiments])
        files = [metric for metric in metrics if config in metric]
        for idx in range(len(files)):
            result = pd.read_parquet(f'metrics/robust/{files[idx]}')
            avg_delta_tmp[:, idx] = get_rolling(result['delta'], window_size)
            std_delta_tmp[:, idx] = get_rolling_std(result['delta'], window_size)

            avg_forced_delta_tmp[:, idx] = get_rolling(result['forced_delta'], window_size)
            std_forced_delta_tmp[:, idx] = get_rolling_std(result['forced_delta'], window_size)

            actions_cols = [col for col in result.columns if col.startswith('actions')]
            for col in actions_cols:
                result[col] = result[col].apply(lambda x: act2prof[int(x)])
            mean_actions = result[actions_cols].mean(axis = 1)
            avg_actions_tmp[:, idx] = get_rolling(mean_actions, window_size)
            std_actions_tmp[:, idx] = get_rolling_std(mean_actions, window_size)

            del result  # Liberar memoria asociada al DataFrame
            
        avg_actions[config] = avg_actions_tmp
        std_actions[config] = std_actions_tmp
        avg_delta[config] = avg_delta_tmp
        std_delta[config] = std_delta_tmp
        avg_forced_delta[config] = avg_forced_delta_tmp
        std_forced_delta[config] = std_forced_delta_tmp

        grouped_configs = ['_'.join(config.split('_')[:-1]) for config in configs]
        grouped_configs = list(set(grouped_configs))
        grouped_configs.remove('bertrand_dqn')
        errorevery = int(0.01 * series_size)

    for config in grouped_configs:
        # average delta
        plt.figure(figsize = (10, 6))
        plt.errorbar(range(series_size), np.mean(avg_delta['bertrand_dqn_base'], axis = 1), np.std(avg_delta['bertrand_dqn_base'], axis = 1), errorevery=errorevery, label = labels[config]['base'])
        plt.errorbar(range(series_size), np.mean(avg_delta[f'{config}_1'], axis = 1), np.std(avg_delta[f'{config}_1'], axis = 1), errorevery=errorevery, label = labels[config]['1'])
        plt.errorbar(range(series_size), np.mean(avg_delta[f'{config}_2'], axis = 1), np.std(avg_delta[f'{config}_2'], axis = 1), errorevery=errorevery, label = labels[config]['2'])
        plt.axhline(0, linestyle = '--', color = 'green', label = 'Monopoly')
        plt.axhline(1, linestyle = '--', color = 'red', label = 'Nash')
        plt.xlabel('Timesteps')
        plt.ylabel('Avg Delta')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'figures/agg_experiments/{config}_delta.pdf')
        plt.close()

        # forced_average delta
        plt.figure(figsize = (10, 6))
        plt.errorbar(range(series_size), np.mean(avg_forced_delta['bertrand_dqn_base'], axis = 1), np.std(avg_forced_delta['bertrand_dqn_base'], axis = 1), errorevery=errorevery, label = labels[config]['base'])
        plt.errorbar(range(series_size), np.mean(avg_forced_delta[f'{config}_1'], axis = 1), np.std(avg_forced_delta[f'{config}_1'], axis = 1), errorevery=errorevery, label = labels[config]['1'])
        plt.errorbar(range(series_size), np.mean(avg_forced_delta[f'{config}_2'], axis = 1), np.std(avg_forced_delta[f'{config}_2'], axis = 1), errorevery=errorevery, label = labels[config]['2'])
        plt.axhline(0, linestyle = '--', color = 'green', label = 'Monopoly')
        plt.axhline(1, linestyle = '--', color = 'red', label = 'Nash')
        plt.xlabel('Timesteps')
        plt.ylabel('Avg Delta')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'figures/agg_experiments/{config}_forced_delta.pdf')
        plt.close()
        
        # std delta
        plt.figure(figsize = (10, 6))
        plt.errorbar(range(series_size), np.mean(std_delta['bertrand_dqn_base'], axis = 1), np.std(std_delta['bertrand_dqn_base'], axis = 1), errorevery=errorevery, label = labels[config]['base'])
        plt.errorbar(range(series_size), np.mean(std_delta[f'{config}_1'], axis = 1), np.std(std_delta[f'{config}_1'], axis = 1), errorevery=errorevery, label = labels[config]['1'])
        plt.errorbar(range(series_size), np.mean(std_delta[f'{config}_2'], axis = 1), np.std(std_delta[f'{config}_2'], axis = 1), errorevery=errorevery, label = labels[config]['2'])
        plt.axhline(0, linestyle = '--', color = 'green', label = 'Fixed $\Delta$')
        plt.xlabel('Timesteps')
        plt.ylabel('Std Delta')
        plt.tight_layout()
        plt.legend()   
        plt.savefig(f'figures/agg_experiments/{config}_std_delta.pdf')
        plt.close()
 

        # std forced delta
        plt.figure(figsize = (10, 6))
        plt.errorbar(range(series_size), np.mean(std_forced_delta['bertrand_dqn_base'], axis = 1), np.std(std_forced_delta['bertrand_dqn_base'], axis = 1), errorevery=errorevery, label = labels[config]['base'])
        plt.errorbar(range(series_size), np.mean(std_forced_delta[f'{config}_1'], axis = 1), np.std(std_forced_delta[f'{config}_1'], axis = 1), errorevery=errorevery, label = labels[config]['1'])
        plt.errorbar(range(series_size), np.mean(std_forced_delta[f'{config}_2'], axis = 1), np.std(std_forced_delta[f'{config}_2'], axis = 1), errorevery=errorevery, label = labels[config]['2'])
        plt.axhline(0, linestyle = '--', color = 'green', label = 'Fixed $\Delta$')
        plt.xlabel('Timesteps')
        plt.ylabel('Std Delta')
        plt.tight_layout()
        plt.legend()  
        plt.savefig(f'figures/agg_experiments/{config}_std_forced_delta.pdf')
        plt.close()

        # avg actions
        plt.figure(figsize = (10, 6))
        plt.errorbar(range(series_size), np.mean(avg_actions['bertrand_dqn_base'], axis = 1), np.std(avg_actions['bertrand_dqn_base'], axis = 1), errorevery=errorevery, label = labels[config]['base'])
        plt.errorbar(range(series_size), np.mean(avg_actions[f'{config}_1'], axis = 1), np.std(avg_actions[f'{config}_1'], axis = 1), errorevery=errorevery, label = labels[config]['1'])
        plt.errorbar(range(series_size), np.mean(avg_actions[f'{config}_2'], axis = 1), np.std(avg_actions[f'{config}_2'], axis = 1), errorevery=errorevery, label = labels[config]['2'])
        plt.axhline(act2prof[0], linestyle = '--', label = 'Lower Bound', color = 'green')
        plt.axhline(act2prof[-1], linestyle = '--', label = 'Upper Bound', color = 'red')
        plt.xlabel('Timesteps')
        plt.ylabel('Avg Price over cost set (%)')
        plt.tight_layout()
        plt.legend()  
        plt.savefig(f'figures/agg_experiments/{config}_actions.pdf')
        plt.close()
        
        # std actions
        plt.figure(figsize = (10, 6))
        plt.errorbar(range(series_size), np.mean(std_actions['bertrand_dqn_base'], axis = 1), np.std(std_actions['bertrand_dqn_base'], axis = 1), errorevery=errorevery, label = labels[config]['base'])
        plt.errorbar(range(series_size), np.mean(std_actions[f'{config}_1'], axis = 1), np.std(std_actions[f'{config}_1'], axis = 1), errorevery=errorevery, label = labels[config]['1'])
        plt.errorbar(range(series_size), np.mean(std_actions[f'{config}_2'], axis = 1), np.std(std_actions[f'{config}_2'], axis = 1), errorevery=errorevery, label = labels[config]['2'])
        plt.xlabel('Timesteps')
        plt.ylabel('Std Price over cost set (%)')
        plt.axhline(0, linestyle = '--', color = 'green', label = 'Fixed Actions')
        plt.tight_layout()
        plt.legend()  
        plt.savefig(f'figures/agg_experiments/{config}_std_actions.pdf')
        plt.close()