import pandas as pd
import numpy as np
import random
import pickle
from agents.dqn import DQNAgent
#from envs.BertrandInflation import BertrandEnv
from envs.BertrandInflation2 import BertrandEnv
from utils.robust_utils import *
import yaml
import os
import itertools
import sys
import torch
import io

def get_intervals(n_intervals = None):
    
    '''
    Returns intervals available of stored models 
    '''
    
    intervals = list(set([model.replace('.pkl', '').split('_')[-1] for model in os.listdir('models/')]))
    intervals.remove('.gitignore')
    intervals = sorted([int(interval) for interval in intervals])

    # Asegurarse de que n_intervals no sea mayor que la longitud de la lista
    if n_intervals:
        n_intervals += 1
        if n_intervals >= len(intervals):
            return intervals
        if n_intervals < 2:
            raise ValueError("n_intervals debe ser al menos 2 para mantener los valores de inicio y término.")

        # Calcular el paso para muestrear la lista, ignorando el primer y último elemento
        step = (len(intervals) - 2) / (n_intervals - 1)
        
        # Construir la lista filtrada
        intervals = [intervals[0]] + [intervals[int(i * step) + 1] for i in range(1, n_intervals - 1)] + [intervals[-1]]

    return intervals[1:]

def get_pairs(n_pairs, n, random_state=None):
    """
    Genera todas las combinaciones posibles de números enteros únicos de tamaño N,
    donde cada número puede tomar un valor entre 0 y K (inclusive).
    Cada número no puede aparecer más de una vez en cada combinación.
    La aleatoriedad en la generación de combinaciones se puede controlar con una semilla.

    Parámetros:
    - n_pairs (int): El tamaño de la muestra
    - n (int): El tamaño de cada combinación (N_agents).
    - k (int): El máximo valor que pueden tomar los números en las combinaciones.
    - random_state (int, opcional): Semilla para controlar la aleatoriedad.

    Retorna:
    - list: Una lista de combinaciones posibles como listas.
    """
    if random_state is not None:
        random.seed(random_state)
        
    # determine nb of experiments
    k = max([int(model.split('_')[3]) for model in os.listdir('models') if model.endswith('.pkl')])

    # Generar los números posibles
    numeros = list(range(1, k + 1))

    # Generar todas las combinaciones posibles sin repetición
    combinaciones = list(itertools.combinations(numeros, n))

    # Mezclar las combinaciones para introducir aleatoriedad
    random.shuffle(combinaciones)

    # Convertir las tuplas a listas para seguir el formato deseado
    combinaciones_lista = [list(combinacion) for combinacion in combinaciones]
    
    # sample
    pairs = random.sample(combinaciones_lista, n_pairs)

    return pairs

def test_agents(env, agents, timesteps = 400_000, exp_name = 'default', n_bins = 4, deviate_step = None):

    '''
    Executes experiment over a test environment. Stores the results on a .parquet file.
    '''

    ob_t = env.reset()
    for timestep in range(1, timesteps + 1):
        actions = [agent.select_action(ob_t, greedy=True) for agent in agents] # select greedy actions

        if deviate_step is not None:
            if timestep == deviate_step:
                env.trigger_deviation = True
            else:
                env.trigger_deviation = False

        ob_t1, rewards, done, info = env.step(actions)
        ob_t = ob_t1

        log = f"\rExperiment: Test_{exp_name} \t Episode completion: {100 * (timestep)/timesteps:.2f} % \t Delta: {info['avg_delta']:.2f} \t Std: {info['std_delta']:.2f} \t Avg Actions: {info['avg_actions']:.2f}"
        sys.stdout.write(log)
        
    delta_history = np.array(env.metric_history)[-timesteps:]
    nash_history = np.array(env.nash_history)[-timesteps:]
    monopoly_history = np.array(env.monopoly_history)[-timesteps:]
    actions_history = np.array(env.action_history)[-timesteps:]
    prices_history = np.array(env.prices_history)[-timesteps:]
    forced_delta_history = np.array(env.forced_metric_history)[-timesteps:]
    IE_history = np.array(env.IE_history)[-timesteps:]
    w_history = np.array(env.w_history)[-timesteps:]

    results = pd.DataFrame({
                            'delta': delta_history,
                            'p_nash': nash_history,
                            'p_monopoly': monopoly_history,
                            'forced_delta': forced_delta_history,
                            'inflation_effect': IE_history,
                            'w': w_history,
                            })

    for agent in range(env.N):
        results[f'actions_{agent}'] = actions_history[:, agent]
        results[f'prices_{agent}'] = prices_history[:, agent]

    # export test metrics
    results.to_parquet(f'metrics/test/{exp_name}.parquet')

    bin_step = timesteps // n_bins

    avg_delta = {str(i): np.mean(results['delta'].iloc[i-bin_step:i]) for i in range(bin_step, timesteps + 1, bin_step)}
    std_delta = {str(i): np.std(results['delta'].iloc[i-bin_step:i]) for i in range(bin_step, timesteps + 1, bin_step)}
    avg_actions = {str(i): np.mean(actions_history[i-bin_step:i]) for i in range(bin_step, timesteps + 1, bin_step)}
    std_actions = {str(i): np.std(actions_history[i-bin_step:i]) for i in range(bin_step, timesteps + 1, bin_step)}
 
    return avg_delta, std_delta, avg_actions, std_actions

def load_parameters(file, device = "cuda:0"):

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)

    if torch.cuda.is_available():
        parameters = pickle.load(file).to(device)
    else:
        parameters = CPU_Unpickler(file).load().to(device)

    return parameters

def get_test_results(configs, test_timesteps = 50_000, n_intervals = 5, n_pairs = 50, 
                     random_state = 3380, device = "cuda:0", deviate_step = None):

    '''
    Executes the experiment on test setting for a determined config and timestep. 
    Stores the results as a .parquet.
    n_pairs: Number of times per config
    n_intervals: Number of points to plot evolution curve
    '''

    intervals = get_intervals(n_intervals = n_intervals) # timesteps of models
    configs = [config.replace('.yaml', '') for config in configs]

    for config in configs: # for each config
        with open(f"configs/{config}.yaml", 'r') as file:
            args = yaml.safe_load(file)
            env_args = args['env']
            dim_actions = args['n_actions']
            dim_states = env_args['N'] * env_args['k'] + env_args['k'] + 1
            
        pairs = get_pairs(n_pairs = n_pairs, n = env_args['N'], random_state = random_state) # idx for each test experiment
        np.random.seed(random_state)
        seeds = np.random.randint(0, 100_000, len(pairs)) # 1 seed for each n° experiment
        
        for interval in intervals: # for each interval
            for pair_idx in range(len(pairs)): # for each combination of agents
                agents = []
                for idx in pairs[pair_idx]: # load models
                    if config == 'bertrand_dqn_deviate':
                        with open(f'models/bertrand_dqn_base_{idx}_{interval}.pkl', 'rb') as file:
                            agent = DQNAgent(dim_states=dim_states, dim_actions=dim_actions, device = device)
                            agent.network = load_parameters(file, device = device)
                            agents.append(agent)
                            deviate_step = args['train']['deviate_step']
                    else:
                        with open(f'models/{config}_{idx}_{interval}.pkl', 'rb') as file:
                            agent = DQNAgent(dim_states=dim_states, dim_actions=dim_actions, device = device)
                            agent.network = load_parameters(file, device = device)
                            agents.append(agent)

                env = BertrandEnv(**env_args, timesteps = test_timesteps, random_state=int(seeds[pair_idx]), dim_actions=dim_actions) 
                exp_name = f'{config}_{interval}_{pair_idx}'

                # test agents and return metrics
                test_agents(env, agents, timesteps = test_timesteps, exp_name=exp_name, deviate_step=deviate_step)

def get_configs():
    
    metrics = os.listdir('metrics/test/')
    
    configs = ['_'.join(metric.split('_')[:4]) if 
                    ('_'.join(metric.split('_')[:3]) != 'bertrand_dqn_base') and 
                    ('_'.join(metric.split('_')[:3]) != 'bertrand_dqn_default') and
                    ('_'.join(metric.split('_')[:3]) != 'bertrand_dqn_deviate') and
                    ('_'.join(metric.split('_')[:3]) != 'bertrand_dqn_altruist') else 
                    '_'.join(metric.split('_')[:3]) for metric in metrics]
    configs.remove('.gitignore')
    configs = list(set(configs))
    
    return configs

def get_test_metrics(test_timesteps = 50_000, n_bins = 4, n_pairs = 50, n_intervals = 5):

    '''
    Builds a dataframe consolidating the test result (AKA test "metric curve") for each config.
    Exports the datamaframe as a .parquet.
    n_pairs: Number of times per config
    n_intervals: Number of points to plot evolution curve
    n_bins: Number of bins to differentiate metric across curve
    '''

    bin_step = test_timesteps // n_bins
    intervals = get_intervals(n_intervals)
    act2prof = np.linspace(-50, 200, 15)

    for config in get_configs():
        avg_delta_curve = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)} 
        std_delta_curve ={str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)}
        avg_forced_delta_curve = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)} 
        std_forced_delta_curve ={str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)}
        avg_actions_curve = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)}
        std_actions_curve = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)}
        for interval in intervals:
            avg_delta_metrics = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)} 
            std_delta_metrics = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)}
            avg_forced_delta_metrics = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)} 
            std_forced_delta_metrics = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)}
            avg_actions_metrics = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)}
            std_actions_metrics = {str(i): [] for i in range(bin_step, test_timesteps + 1, bin_step)}
            for pair in range(n_pairs):
                tmp = pd.read_parquet(f'metrics/test/{config}_{interval}_{pair}.parquet') # read file

                # TRANSFORM ACTIONS TO PERCENT
                actions_cols = [col for col in tmp.columns if 'actions' in col]
                for col in actions_cols:
                    tmp[col] = tmp[col].apply(lambda x: act2prof[x])

                avg_delta = {str(i): np.mean(tmp['delta'].iloc[i-bin_step:i]) for i in range(bin_step, test_timesteps + 1, bin_step)}
                std_delta = {str(i): np.std(tmp['delta'].iloc[i-bin_step:i]) for i in range(bin_step, test_timesteps + 1, bin_step)}
                avg_forced_delta = {str(i): np.mean(tmp['forced_delta'].iloc[i-bin_step:i]) for i in range(bin_step, test_timesteps + 1, bin_step)}
                std_forced_delta = {str(i): np.std(tmp['forced_delta'].iloc[i-bin_step:i]) for i in range(bin_step, test_timesteps + 1, bin_step)}
                avg_actions = {str(i): np.mean(tmp[actions_cols][i-bin_step:i]) for i in range(bin_step, test_timesteps + 1, bin_step)}
                std_actions = {str(i): np.std(tmp[actions_cols][i-bin_step:i]) for i in range(bin_step, test_timesteps + 1, bin_step)}

                # store metrics
                for key in avg_delta_metrics.keys():
                    avg_delta_metrics[key].append(avg_delta[key])
                    std_delta_metrics[key].append(std_delta[key])
                    avg_forced_delta_metrics[key].append(avg_forced_delta[key])
                    std_forced_delta_metrics[key].append(std_forced_delta[key])
                    avg_actions_metrics[key].append(avg_actions[key])
                    std_actions_metrics[key].append(std_actions[key])

            # obtain aggregate statistics
            for key in avg_delta_curve.keys():
                avg_delta_curve[key].append(np.mean(avg_delta_metrics[key])) # average of average delta per timestep
                std_delta_curve[key].append(np.std(avg_delta_metrics[key])) # std of average delta per timestep
                avg_forced_delta_curve[key].append(np.mean(avg_forced_delta_metrics[key])) # average of average forced delta per timestep
                std_forced_delta_curve[key].append(np.std(avg_forced_delta_metrics[key])) # std of average forced delta per timestep
                avg_actions_curve[key].append(np.mean(avg_actions_metrics[key])) # average of average action per timestep
                std_actions_curve[key].append(np.std(avg_actions_metrics[key])) # std of average action per timestep

        # export results of config
        results = pd.DataFrame({
        'interval': intervals,
        })

        for key in avg_actions_curve.keys():
            results[f'avg_delta_{key}'] = avg_delta_curve[key]
            results[f'std_delta_{key}'] = std_delta_curve[key]
            results[f'avg_forced_delta_{key}'] = avg_forced_delta_curve[key]
            results[f'std_forced_delta_{key}'] = std_forced_delta_curve[key]
            results[f'avg_actions_{key}'] = avg_actions_curve[key]
            results[f'std_actions_{key}'] = std_actions_curve[key]

        results.to_parquet(f'metrics/test/{config}_curve.parquet')

def get_test_tables(nb_experiments = 50):
    metrics = os.listdir('metrics/test/')
    configs = get_configs()

    robust_delta = {} # dict with list of deltas for each config
    robust_forced_delta = {} # dict with list of forced_deltas for each config
    for config in configs:

        timesteps = list(set([int(model.split('_')[-1].replace('.pkl', '')) for model in os.listdir('models/') if model != '.gitignore']))
        max_timestep = max(timesteps)

        files = [metric.replace('.parquet', '') for metric in metrics if config in metric and str(max_timestep) in metric]
        files = sorted(files, key=lambda x: int(x.split('_')[-1])) # sort by experiment idx
        assert len(files) == nb_experiments, f'error calculating sample of experiment {config}: {len(files)} experiments' # assert of nb of experiments
        
        deltas = []
        forced_deltas = []
        for file in files:
            result = pd.read_parquet(f'metrics/test/{file}.parquet')
            #df_tail = result.iloc[-evaluation_timesteps:]
            df_tail = result.copy()

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

    delta_single_statistics.to_csv('tables/test_delta_single_statistics.txt', sep = '|', encoding = 'utf-8-sig')
    delta_comparison_statistics.to_csv('tables/test_delta_comparison_statistics.txt', sep = '|', encoding = 'utf-8-sig')

    forced_delta_single_statistics.to_csv('tables/test_forced_delta_single_statistics.txt', sep = '|', encoding = 'utf-8-sig')
    forced_delta_comparison_statistics.to_csv('tables/test_forced_delta_comparison_statistics.txt', sep = '|', encoding = 'utf-8-sig')