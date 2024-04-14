import sys
import numpy as np
import pandas as pd
import pickle

def train(env, agents, buffer, N, episodes, timesteps, update_steps, variation, 
          deviate_step, test_size, exp_name = 'experiment', 
          debug = False, robust = False, store_steps = 10_000):

    '''
    Performs a rollout of an experiment. Trains agents and store results as a csv.
    env: environment of experiment
    agents: list of agents to play the environment
    buffer: replay buffer
    N: number of agents
    episodes: number of episodes
    timesteps: number of timesteps of each experiment
    update_steps: number of steps to perform update
    variation: variation of the experiment
    deviate_start: percent of experiment to start the deviation
    deviate_end: percent of experiment to end the deviation
    test_size: test size to perform actions without update
    '''
    
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
    epsilon_history = np.zeros((episodes, timesteps))
    forced_delta_history = np.zeros((episodes, timesteps))
    w_history = np.zeros((episodes, timesteps))
    IE_history = np.zeros((episodes, timesteps))
    
    for episode in range(episodes):
        ob_t = env.reset()
        update_agents = True
        for t in range(1, timesteps + 1):
            actions = [agent.select_action(ob_t) for agent in agents]    
            
            if variation == 'deviate':
                if t == deviate_step:
                    env.trigger_deviation = True
                else:
                    env.trigger_deviation = False
            
            elif variation == 'altruist':
                env.altruist = True
            
            elif (variation == 'train-test'):
                if (t/timesteps > (1 - test_size)):
                    update_agents = False
            
            ob_t1, rewards, done, info = env.step(actions)
            
            experience = (ob_t, actions, rewards, ob_t1, done)
            
            buffer.store_transition(*experience)
            
            if (t % update_steps == 0) & (t >= buffer.sample_size) & (update_agents):
                for agent_idx in range(N):
                    agent = agents[agent_idx]
                    sample = buffer.sample(agent_idx)
                    agent.update(*sample)

            if (t % store_steps == 0) & (variation != 'deviate'):
                # save agent
                with open(f'models/{exp_name}_{t}.pkl', 'wb') as file:
                    pickle.dump(agents[0].network, file)
            
            log = f"\rExperiment: Train_{exp_name} \t Episode: {episode + 1}/{episodes} \t Episode completion: {100 * (t+1)/timesteps:.2f} % \t Delta: {info['avg_delta']:.2f} \t Std: {info['std_delta']:.2f}"
            if debug:
                epsilon = np.mean(env.epsilon_history[-1000:])
                log += f"\t Epsilon: {epsilon:.2f}"
            sys.stdout.write(log)
            
            ob_t = ob_t1
        
        # store episode metrics
        if debug:
            quantities_history[episode] = np.array(env.quantities_history)[-timesteps:]
            pi_N_history[episode] = np.array(env.pi_N_history)[-timesteps:]
            pi_M_history[episode] = np.array(env.pi_M_history)[-timesteps:]
            A_history[episode] = np.array(env.A_history)[-timesteps:]
            rewards_history[episode] = np.array(env.rewards_history)[-timesteps:]
            try:
                epsilon_history[episode] = np.array(agent.epsilon_history)[-timesteps:]
            except:
                epsilon_history[episode] = np.zeros(timesteps)

        prices_history[episode] = np.array(env.prices_history)[-timesteps:]
        actions_history[episode] = np.array(env.action_history)[-timesteps:]
        delta_history[episode] = np.array(env.metric_history)[-timesteps:]
        monopoly_history[episode] = np.array(env.monopoly_history)[-timesteps:]
        nash_history[episode] = np.array(env.nash_history)[-timesteps:]
        costs_history[episode] = np.array(env.costs_history)[-timesteps:]
        forced_delta_history[episode] = np.array(env.forced_metric_history)[-timesteps:]
        IE_history[episode] = np.array(env.IE_history)[-timesteps:]
        w_history[episode] = np.array(env.w_history)[-timesteps:]

    # export   
    if debug:
        rewards_history = np.mean(rewards_history, axis = 0)
        quantities_history = np.mean(quantities_history, axis = 0)
        pi_N_history = np.mean(pi_N_history, axis = 0)
        pi_M_history = np.mean(pi_M_history, axis = 0)
        A_history = np.mean(A_history, axis = 0) # equal disposition to pay
        epsilon_history = np.mean(epsilon_history, axis = 0)

    prices_history = np.mean(prices_history, axis = 0)
    actions_history = np.mean(actions_history, axis = 0)
    monopoly_history = np.mean(monopoly_history, axis = 0)
    nash_history = np.mean(nash_history, axis = 0)
    delta_history = np.mean(delta_history, axis = 0)
    costs_history = np.mean(costs_history, axis = 0)
    forced_delta_history = np.mean(forced_delta_history, axis = 0)
    IE_history = np.mean(IE_history, axis = 0)
    w_history = np.mean(w_history, axis = 0)
        
    if debug:
        results = pd.DataFrame({'costs': costs_history,
                                'pi_N': pi_N_history,
                                'pi_M': pi_M_history,
                                'delta': delta_history,
                                'p_nash': nash_history,
                                'p_monopoly': monopoly_history,
                                'A': A_history,
                                'epsilon': epsilon_history,
                                })
        
    else:
        results = pd.DataFrame({
                                'delta': delta_history,
                                'p_nash': nash_history,
                                'p_monopoly': monopoly_history,
                                'costs': costs_history,
                                'forced_delta': forced_delta_history,
                                'inflation_effect': IE_history,
                                'w': w_history,
                                })

    for agent in range(env.N):
        results[f'actions_{agent}'] = actions_history[:, agent]
        results[f'prices_{agent}'] = prices_history[:, agent]
        if debug:
            results[f'quantities_{agent}'] = quantities_history[:, agent]
            results[f'rewards_{agent}'] = rewards_history[:, agent]
    
    save_path = f'metrics/robust/{exp_name}.parquet' if robust else f'metrics/single/{exp_name}.parquet'
    results.to_parquet(save_path, index = False)