from agents.dqn import DQNAgent
from agents.sac import SACAgent
from agents.ddpg import DDPGAgent
from envs.LinearBertrandInflation import LinearBertrandEnv
#from envs.BertrandInflation import BertrandEnv
from envs.BertrandInflation2 import BertrandEnv
from replay_buffer import ReplayBuffer
import numpy as np
from tqdm import tqdm
import sys
import pickle
import matplotlib.pyplot as plt
from utils.get_plots import get_rolling, get_rolling_std
import yaml
import pandas as pd
import os

def train_seed_agents(seeds: list):
    
    for random_state in seeds:

        envs_dict = {'bertrand': BertrandEnv, 'linear': LinearBertrandEnv}
        models_dict = {'sac': SACAgent, 'ddpg': DDPGAgent, 'dqn': DQNAgent}

        with open(f"configs/bertrand_dqn_base.yaml", 'r') as file:
            args = yaml.safe_load(file)
            agent_args = args['agent']
            env_args = args['env']
            buffer_args = args['buffer']
            train_args = args['train']

        np.random.seed(random_state)

        # dimensions
        dim_states = (env_args['N'] * env_args['k']) + env_args['k'] + 1
        dim_actions = args['n_actions'] if args['model'] == 'dqn' else 1
        
        # load environment, agent and buffer
        env = envs_dict[args['env_name']]
        env = env(**env_args, timesteps = train_args['timesteps'], dim_actions = dim_actions, random_state = random_state)      
        
        model = models_dict[args['model']] 
        agents = [model(dim_states, dim_actions, **agent_args, random_state = random_state + _) for _ in range(env.N)]
        buffer = ReplayBuffer(dim_states = dim_states, N = env.N, **buffer_args)

        exp_name = f'agent_{random_state}'
        episodes = 1
        timesteps = train_args['timesteps']
        #timesteps = 2000
        update_steps = train_args['update_steps']
        N = env.N

        prices_history = np.zeros((episodes, timesteps, N))
        actions_history = np.zeros((episodes, timesteps, N))
        costs_history = np.zeros((episodes, timesteps))
        monopoly_history = np.zeros((episodes, timesteps))
        nash_history = np.zeros((episodes, timesteps))
        rewards_history = np.zeros((episodes, timesteps, N))
        delta_history = np.zeros((episodes, timesteps))
        quantities_history = np.zeros((episodes, timesteps, N))
        pi_N_history = np.zeros((episodes, timesteps))
        pi_M_history = np.zeros((episodes, timesteps))
        A_history = np.zeros((episodes, timesteps))

        ob_t = env.reset()
        for episode in range(episodes):
            for t in range(timesteps):
                actions = [agent.select_action(ob_t) for agent in agents]
                
                ob_t1, rewards, done, info = env.step(actions)
                
                experience = (ob_t, actions, rewards, ob_t1, done)
                
                buffer.store_transition(*experience)
                
                if (t % update_steps == 0) & (t >= buffer.sample_size):
                    for agent_idx in range(N):
                        agent = agents[agent_idx]
                        sample = buffer.sample(agent_idx)
                        agent.update(*sample)
                        
                log = f"\rExperiment: {exp_name} \t Episode: {episode + 1}/{episodes} \t Episode completion: {100 * t/timesteps:.2f} % \t Delta: {info['avg_delta']:.2f} \t Std: {info['std_delta']:.2f}"
                try:
                    epsilon = np.mean(agent.epsilon_history[-1000:])
                    log += f"\t Epsilon: {epsilon:.2f}"
                except:
                    pass
                sys.stdout.write(log)
                        
                ob_t = ob_t1
                
            # store episode metrics
            prices_history[episode] = np.array(env.prices_history)[-timesteps:]
            actions_history[episode] = np.array(env.action_history)[-timesteps:]
            costs_history[episode] = np.array(env.costs_history)[-timesteps:]
            monopoly_history[episode] = np.array(env.monopoly_history)[-timesteps:]
            nash_history[episode] = np.array(env.nash_history)[-timesteps:]
            rewards_history[episode] = np.array(env.rewards_history)[-timesteps:]
            delta_history[episode] = np.array(env.metric_history)[-timesteps:]
            quantities_history[episode] = np.array(env.quantities_history)[-timesteps:]
            pi_N_history[episode] = np.array(env.pi_N_history)[-timesteps:]
            pi_M_history[episode] = np.array(env.pi_M_history)[-timesteps:]
            A_history[episode] = np.array(env.A_history)[-timesteps:]

        prices_history = np.mean(prices_history, axis = 0)
        actions_history = np.mean(actions_history, axis = 0)
        costs_history = np.mean(costs_history, axis = 0)
        monopoly_history = np.mean(monopoly_history, axis = 0)
        nash_history = np.mean(nash_history, axis = 0)
        rewards_history = np.mean(rewards_history, axis = 0)
        delta_history = np.mean(delta_history, axis = 0)
        quantities_history = np.mean(quantities_history, axis = 0)
        pi_N_history = np.mean(pi_N_history, axis = 0)
        pi_M_history = np.mean(pi_M_history, axis = 0)
        A_history = np.mean(A_history, axis = 0) # equal disposition to pay
        
        # save metrics
        results = pd.DataFrame({'costs': costs_history,
                        'pi_N': pi_N_history,
                        'pi_M': pi_M_history,
                        'delta': delta_history,
                        'p_nash': nash_history,
                        'p_monopoly': monopoly_history,
                        'A': A_history,
                        })

        for agent in range(env.N):
            results[f'actions_{agent}'] = actions_history[:, agent]
            results[f'prices_{agent}'] = prices_history[:, agent]
            results[f'quantities_{agent}'] = quantities_history[:, agent]
            results[f'rewards_{agent}'] = rewards_history[:, agent]
            
        results.to_csv(f'metrics/{exp_name}.csv', index = False, sep = ';', encoding = 'utf-8-sig')

        # save agent
        with open(f'models/agent_{random_state}.pkl', 'wb') as file:
            pickle.dump(agents[0], file)
        
def test_seed_agents(seeds: list, random_state: int = 500):

    envs_dict = {'bertrand': BertrandEnv, 'linear': LinearBertrandEnv}

    with open(f"configs/bertrand_dqn_base.yaml", 'r') as file:
        args = yaml.safe_load(file)
        env_args = args['env']
        train_args = args['train']

    np.random.seed(random_state)

    # dimensions
    dim_states = (env_args['N'] * env_args['k']) + env_args['k'] + 1
    dim_actions = args['n_actions'] if args['model'] == 'dqn' else 1

    # load environment, agent and buffer
    env = envs_dict[args['env_name']]
    env = env(**env_args, timesteps = train_args['timesteps'], dim_actions = dim_actions, random_state = random_state)      
    
    # load trained agents
    agents = []
    for seed in seeds:
        with open(f'models/agent_{seed}.pkl', 'rb') as file:
            agent = pickle.load(file)
            agents.append(agent)

    exp_name = 'test_agents'
    episodes = 1
    timesteps = train_args['timesteps']
    N = env.N

    prices_history = np.zeros((episodes, timesteps, N))
    actions_history = np.zeros((episodes, timesteps, N))
    costs_history = np.zeros((episodes, timesteps))
    monopoly_history = np.zeros((episodes, timesteps))
    nash_history = np.zeros((episodes, timesteps))
    rewards_history = np.zeros((episodes, timesteps, N))
    delta_history = np.zeros((episodes, timesteps))
    quantities_history = np.zeros((episodes, timesteps, N))
    pi_N_history = np.zeros((episodes, timesteps))
    pi_M_history = np.zeros((episodes, timesteps))
    A_history = np.zeros((episodes, timesteps))

    ob_t = env.reset()
    for episode in range(episodes):
        for t in range(timesteps):
            # frozen gradients
            actions = [agent.select_action(ob_t, greedy = True) for agent in agents]
            
            ob_t1, rewards, done, info = env.step(actions)
            
            log = f"\rExperiment: {exp_name} \t Episode: {episode + 1}/{episodes} \t Episode completion: {100 * t/timesteps:.2f} % \t Delta: {info['avg_delta']:.2f} \t Std: {info['std_delta']:.2f}"
            try:
                epsilon = np.mean(agent.epsilon_history[-1000:])
                log += f"\t Epsilon: {epsilon:.2f}"
            except:
                pass
            sys.stdout.write(log)
                    
            ob_t = ob_t1
            
        # store episode metrics
        prices_history[episode] = np.array(env.prices_history)[-timesteps:]
        actions_history[episode] = np.array(env.action_history)[-timesteps:]
        costs_history[episode] = np.array(env.costs_history)[-timesteps:]
        monopoly_history[episode] = np.array(env.monopoly_history)[-timesteps:]
        nash_history[episode] = np.array(env.nash_history)[-timesteps:]
        rewards_history[episode] = np.array(env.rewards_history)[-timesteps:]
        delta_history[episode] = np.array(env.metric_history)[-timesteps:]
        quantities_history[episode] = np.array(env.quantities_history)[-timesteps:]
        pi_N_history[episode] = np.array(env.pi_N_history)[-timesteps:]
        pi_M_history[episode] = np.array(env.pi_M_history)[-timesteps:]
        A_history[episode] = np.array(env.A_history)[-timesteps:]

    prices_history = np.mean(prices_history, axis = 0)
    actions_history = np.mean(actions_history, axis = 0)
    costs_history = np.mean(costs_history, axis = 0)
    monopoly_history = np.mean(monopoly_history, axis = 0)
    nash_history = np.mean(nash_history, axis = 0)
    rewards_history = np.mean(rewards_history, axis = 0)
    delta_history = np.mean(delta_history, axis = 0)
    quantities_history = np.mean(quantities_history, axis = 0)
    pi_N_history = np.mean(pi_N_history, axis = 0)
    pi_M_history = np.mean(pi_M_history, axis = 0)
    A_history = np.mean(A_history, axis = 0) # equal disposition to pay
    
    # save metrics
    results = pd.DataFrame({'costs': costs_history,
                    'pi_N': pi_N_history,
                    'pi_M': pi_M_history,
                    'delta': delta_history,
                    'p_nash': nash_history,
                    'p_monopoly': monopoly_history,
                    'A': A_history,
                    })

    for agent in range(env.N):
        results[f'actions_{agent}'] = actions_history[:, agent]
        results[f'prices_{agent}'] = prices_history[:, agent]
        results[f'quantities_{agent}'] = quantities_history[:, agent]
        results[f'rewards_{agent}'] = rewards_history[:, agent]
        
    results.to_csv(f'metrics/{exp_name}.csv', index = False, sep = ';', encoding = 'utf-8-sig')

def train_test_seed_agents(n_agents: int = 2, random_state: int = 3381):
    
    agent_seeds = [random_state + i for i in range(n_agents)]
    
    train_seed_agents(seeds = agent_seeds)
    test_seed_agents(seeds = agent_seeds)
    
    
def plot_train_test(window_size: int = 1000, figsize = (6, 4)):
    
    ## PLOT TRAIN

    agent_metrics = [metric for metric in os.listdir('metrics/') if 'agent_' in metric]

    for metric in agent_metrics:

        df_metric = pd.read_csv(f'metrics/{metric}', sep = ';')
        random_state = metric.split('_')[1].replace('.csv', '')

        window_size = 1000

        prices_serie = get_rolling(df_metric['prices_0'], window_size)
        prices_serie_std = get_rolling_std(df_metric['prices_0'], window_size)
        monopoly_history = df_metric['p_monopoly']
        nash_history = df_metric['p_nash']

        series_size = len(prices_serie)

        plt.figure(figsize = figsize)
        plt.errorbar(range(series_size), prices_serie, prices_serie_std, errorevery=int(0.01 * series_size), label = f'Agent 0')
        plt.plot(monopoly_history, color = 'red', label = 'Monopoly')
        plt.plot(nash_history, color = 'green', label = 'Nash')
        plt.xlabel('Timesteps')
        plt.ylabel('Prices')
        plt.legend()
        plt.savefig(f'figures/simple_experiments/bertrand_dqn_agent-{random_state}.pdf')
    
    
    ## PLOT TEST
    
    df_avg = pd.DataFrame()
    df_std = pd.DataFrame()
    
    df_plot = pd.read_csv('metrics/test_agents.csv', sep = ';', encoding = 'utf-8-sig')
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

    # plot
    plt.figure(figsize = figsize)
    for agent in range(n_agents):
        serie = f'prices_{agent}'
        plt.errorbar(range(series_size), df_avg[serie], df_std[serie], errorevery=int(0.01 * series_size), label = f'Agent {agent}')
    plt.plot(df_plot['p_monopoly'], color = 'red', label = 'Monopoly price')
    plt.plot(df_plot['p_nash'], color = 'green', label = 'Nash price')
    plt.xlabel('Timesteps')
    plt.ylabel('Prices')
    #plt.title('Experiments Results Sample')
    plt.legend(loc = 'lower right')
    plt.savefig('figures/simple_experiments/bertrand_dqn_separate-train-test_1_prices.pdf')
    

    plt.figure(figsize = figsize)
    plt.errorbar(range(series_size), df_avg['delta'], df_std['delta'], errorevery=int(0.01 * series_size), label = f'Average profits')
    plt.axhline(1, color = 'red', label = 'Nash')
    plt.axhline(0, color = 'green', label = 'Monopoly')
    plt.xlabel('Timesteps')
    plt.ylabel('Delta')
    #plt.title('Experiments Results Sample')
    plt.legend(loc = 'lower right')
    plt.savefig('figures/simple_experiments/bertrand_dqn_separate-train-test_1_delta.pdf')