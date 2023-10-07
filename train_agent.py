import numpy as np
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from utils.parse_args import parse_args
from utils.plot_metrics import plot_metrics
from utils.export_results import export_results
import sys
import pandas as pd

from envs.BertrandInflation import BertrandEnv
from envs.LinearBertrandInflation import LinearBertrandEnv

from agents.ddpg import DDPGAgent
from agents.dqn import DQNAgent
from agents.sac import SACAgent

models_dict = {'sac': SACAgent, 'ddpg': DDPGAgent, 'dqn': DQNAgent}
envs_dict = {'bertrand': BertrandEnv, 'linear': LinearBertrandEnv}

if __name__ == '__main__':
    
    # load args
    args = parse_args()

    # initiate environment
    env = envs_dict[args['env']]
    env = env(N = args['N'], k = args['k'], rho = args['rho'], v = args['v'], xi = args['xi'], timesteps = args['timesteps'], inflation_step = args['inflation_step'])
    
    # get dimensions
    #dim_states = env.N if args['use_lstm'] else env.k * env.N + env.k + 1
    dim_states = (args['N'] * args['k']) + (args['k'] + 1 ) * 2 + args['N']
    dim_actions = args['n_actions'] if args['model'] == 'dqn' else 1

    # initiate agents
    model = models_dict[args['model']]
    agents = [model(dim_states, dim_actions) for _ in range(args['N'])]
    
    # initiate buffer
    buffer = ReplayBuffer(dim_states = dim_states, N = args['N'], buffer_size = args['buffer_size'], sample_size = args['sample_size'])

    prices_history = np.zeros((args['episodes'], args['timesteps'], args['N']))
    actions_history = np.zeros((args['episodes'], args['timesteps'], args['N']))
    monopoly_history = np.zeros((args['episodes'], args['timesteps']))
    nash_history = np.zeros((args['episodes'], args['timesteps']))
    rewards_history = np.zeros((args['episodes'], args['timesteps'], args['N']))
    metric_history = np.zeros((args['episodes'], args['timesteps']))

    # train
    for episode in range(args['episodes']):
        ob_t = env.reset()
        # initiate plot
        plot_dim = (2, 3) if args['plot_loss'] else (1, 3)
        fig, axes = plt.subplots(*plot_dim, figsize = (16, 6) if args['plot_loss'] else (16, 4))
        axes = np.array(axes, ndmin = 2)
        for t in range(args['timesteps']):
            # select action
            actions = [agent.select_action(ob_t) for agent in agents] 
            
            # trigger deviation
            #if (t > args['timesteps'] // 2) & (args['trigger_deviation']):
            #    actions[0] = env.pN
            
            #actions = [env.pM for _ in range(env.N)]
            
            # step
            ob_t1, rewards, done, info = env.step(actions)
            
            # store transition
            experience = (ob_t, actions, rewards, ob_t1, done)
            buffer.store_transition(*experience)
            
            # update and plot
            if (t % args['update_steps'] == 0) & (t >= args['sample_size']):
                # update
                for agent_idx in range(args['N']):
                    agent = agents[agent_idx]
                    sample = buffer.sample(agent_idx)
                    agent.update(*sample)

                # plot
                if t % args['plot_steps'] == 0:
                    plot_args = (fig, axes, env.prices_history, env.monopoly_history, env.nash_history, env.rewards_history, env.metric_history, args['window'])
                    plot_metrics(*plot_args, agent.actor_loss, agent.Q_loss) if args['plot_loss'] else plot_metrics(*plot_args)     
            
            sys.stdout.write(f"\rEpisode: {episode + 1}/{args['episodes']} \t Training completion: {100 * t/args['timesteps']:.2f} % \t Delta: {info:.2f}")
            
            # update ob_t
            ob_t = ob_t1
            
        # store metrics
        prices_history[episode] = np.array(env.prices_history)[args['k']:]
        monopoly_history[episode] = np.array(env.monopoly_history)
        nash_history[episode] = np.array(env.nash_history)
        rewards_history[episode] = np.array(env.rewards_history)
        metric_history[episode] = np.array(env.metric_history)
    
    # save plot
    #plt.savefig(f"figures/{args['exp_name']}.pdf")
    
    # export results
    #export_results(env.prices_history[env.k:], env.quantities_history,
    #               env.monopoly_history[1:], env.nash_history[1:], 
    #               env.rewards_history, env.metric_history, 
    #               env.pi_N_history[1:], env.pi_M_history[1:],
    #               env.costs_history[env.v+1:], args['exp_name'])
    
    # export
    prices_history = np.mean(prices_history, axis = 0)
    actions_history = np.mean(actions_history, axis = 0)
    monopoly_history = np.mean(monopoly_history, axis = 0)
    nash_history = np.mean(nash_history, axis = 0)
    rewards_history = np.mean(rewards_history, axis = 0)
    metric_history = np.mean(metric_history, axis = 0)
    
    results = pd.DataFrame({'monopoly': monopoly_history,
                        'nash': nash_history,
                        'metric': metric_history
                        })

    for agent in range(env.N):
        results[f'prices_{agent}'] = prices_history[:, agent]
        results[f'actions_{agent}'] = actions_history[:, agent]
        results[f'rewards_{agent}'] = rewards_history[:, agent]
        
    results.to_csv('metrics/experiment.csv')