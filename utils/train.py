import sys
import numpy as np
import pandas as pd

def train(env, agents, buffer, N, episodes, timesteps, update_steps, variation, deviate_start, deviate_end, test_size, exp_name = 'experiment'):
    
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
    
    for episode in range(episodes):
        ob_t = env.reset()
        update_agents = True
        for t in range(timesteps):
            actions = [agent.select_action(ob_t) for agent in agents]    
            
            if variation == 'deviate':
                if (t/timesteps > deviate_start) and (t/timesteps <= deviate_end):
                    env.trigger_deviation = True
                
                elif t/timesteps > deviate_end:
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
            
            sys.stdout.write(f"\rExperiment: {exp_name} \t Episode: {episode + 1}/{episodes} \t Episode completion: {100 * t/timesteps:.2f} % \t Delta: {info:.2f}")
            
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
    
    # export   
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