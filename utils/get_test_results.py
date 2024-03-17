import pandas as pd
import numpy as np
import random
import pickle
from agents.dqn import DQNAgent
from envs.BertrandInflation import BertrandEnv
from utils.robust_utils import *
from tqdm import tqdm
import yaml
import os
import itertools

def get_intervals(n_intervals = None):
    
    '''
    Returns intervals available of stored models 
    '''
    
    intervals = list(set([model.replace('.pkl', '').split('_')[-1] for model in os.listdir('models/')]))
    intervals.remove('.gitignore')
    intervals = sorted([int(interval) for interval in intervals])

    # Asegurarse de que n_intervals no sea mayor que la longitud de la lista
    if n_intervals:
        if n_intervals >= len(intervals):
            return intervals
        if n_intervals < 2:
            raise ValueError("n_intervals debe ser al menos 2 para mantener los valores de inicio y término.")

        # Calcular el paso para muestrear la lista, ignorando el primer y último elemento
        step = (len(intervals) - 2) / (n_intervals - 1)
        
        # Construir la lista filtrada
        intervals = [intervals[0]] + [intervals[int(i * step) + 1] for i in range(1, n_intervals - 1)] + [intervals[-1]]

    return intervals

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

def test_agents(env, agents, timesteps = 10_000):

    ob_t = env.reset()
    for timestep in range(timesteps):
        
        actions = [agent.select_action(ob_t, greedy=True) for agent in agents] # select greedy actions
        
        ob_t1, rewards, done, info = env.step(actions)
        
        ob_t = ob_t1
        
    delta_history = np.array(env.metric_history)[-timesteps:]
    nash_history = np.array(env.nash_history)[-timesteps:]
    monopoly_history = np.array(env.monopoly_history)[-timesteps:]
    actions_history = np.array(env.action_history)[-timesteps:]
    prices_history = np.array(env.prices_history)[-timesteps:]

    results = pd.DataFrame({
                            'delta': delta_history,
                            'p_nash': nash_history,
                            'p_monopoly': monopoly_history,
                            })

    for agent in range(env.N):
        results[f'actions_{agent}'] = actions_history[:, agent]
        results[f'prices_{agent}'] = prices_history[:, agent]
        
    return np.mean(delta_history), np.std(delta_history), np.mean(actions_history), np.std(delta_history)

def get_test_results(test_timesteps = 50_000, n_intervals = None, n_pairs = 50, random_state = 3380):

    intervals = get_intervals(n_intervals = n_intervals) # timesteps of models
    configs = get_configs() # configs 

    for config in tqdm(configs): # for each config
        with open(f"configs/{config}.yaml", 'r') as file:
            args = yaml.safe_load(file)
            env_args = args['env']
            dim_actions = args['n_actions']
            dim_states = env_args['N'] * env_args['k'] + env_args['k'] + 1
            
        pairs = get_pairs(n_pairs = n_pairs, n = env_args['N'], random_state = random_state) # idx for each test experiment
        np.random.seed(random_state)
        seeds = np.random.randint(0, 100_000, len(pairs))
        
        avg_delta_curve, std_delta_curve = [], []
        avg_actions_curve, std_actions_curve = [], []
        for interval in intervals: # for each interval
            avg_delta_list, std_delta_list = [], []
            avg_actions_list, std_actions_list = [], []
            for pair_idx in range(len(pairs)): # for each combination of agents
                agents = []
                for idx in pairs[pair_idx]: # load models
                    with open(f'models/{config}_{idx}_{interval}.pkl', 'rb') as file:
                        agent = DQNAgent(dim_states=dim_states, dim_actions=dim_actions)
                        agent.network = pickle.load(file)
                        agents.append(agent)

                env = BertrandEnv(**env_args, timesteps = test_timesteps, random_state=int(seeds[pair_idx])) 

                # test agents and return metrics
                avg_delta, std_delta, avg_actions, std_actions = test_agents(env, agents, timesteps = test_timesteps)

                # store metrics
                avg_delta_list.append(avg_delta)
                std_delta_list.append(std_delta)
                avg_actions_list.append(avg_actions)
                std_actions_list.append(std_actions)
            
            # obtain aggregate statistics
            avg_delta_curve.append(np.mean(avg_delta_list)) # average of average delta per timestep
            std_delta_curve.append(np.std(avg_delta_list)) # std of average delta per timestep
            avg_actions_curve.append(np.mean(avg_actions_list)) # average of average action per timestep
            std_actions_curve.append(np.std(avg_actions_list)) # std of average action per timestep

        # export results of config
        results = pd.DataFrame({
            'interval': intervals,
            'avg_delta': avg_actions_curve,
            'std_delta': std_delta_curve,
            'avg_actions': avg_actions_curve,
            'std_actions': std_actions_curve,
            })

        results.to_parquet(f'metrics/test/{config}.parquet')