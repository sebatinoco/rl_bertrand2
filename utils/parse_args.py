import argparse
from distutils.util import strtobool

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # env arguments
    parser.add_argument('--env', type = str, default = 'bertrand', help = 'environment to run the experiment')
    parser.add_argument('--N', type = int, default = 2, help = 'number of agents')
    parser.add_argument('--k', type = int, default = 1, help = 'past periods observed by agents')
    parser.add_argument('--v', type = int, default = 3, help = 'past periods to predict next inflation value')
    parser.add_argument('--rho', type = float, default = 0.001, help = 'probability of changing prices')
    parser.add_argument('--xi', type = float, default = 0.2, help = 'term to amplify range of actions')
    
    # buffer arguments
    parser.add_argument('--sample_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--buffer_size', type = int, default = 200000, help = 'buffer size')
    
    # agent arguments
    parser.add_argument('--model', type = str, default = 'sac', help = 'agent model')
    parser.add_argument('--actor_lr', type = float, default = 0.01, help = 'learning rate of the agents')
    parser.add_argument('--Q_lr', type = float, default = 0.01, help = 'learning rate of the agents')
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'gamma coeff of the agents')
    parser.add_argument('--tau', type = float, default = 0.001, help = 'tau coeff of the agents')
    parser.add_argument('--hidden_size', type = int, default = 256, help = 'hidden dim of the agents')
    parser.add_argument('--inflation_step', type = int, default = 0, help = 'step to start inflation')
    
    # train arguments
    parser.add_argument('--episodes', type = int, default = 10, help = 'number of episodes')
    parser.add_argument('--timesteps', type = int, default = 2000, help = 'number of steps')
    parser.add_argument('--learning_start', type = int, default = 100, help = 'steps to start learning')
    parser.add_argument('--update_steps', type = int, default = 1, help = 'steps per update')
    parser.add_argument('--plot_steps', type = int, default = 50, help = 'steps per update')
    parser.add_argument('--window', type = int, default = 1000, help = 'rolling steps')
    parser.add_argument('--use_lstm', type = lambda x: bool(strtobool(x)), default = False, help = 'enable lstm')
    parser.add_argument('--n_actions', default = 15, help = 'number of actions', type = int)
    parser.add_argument('--inflation_start', type = int, default = 1000, help = 'min steps to price changes')
    parser.add_argument('--trigger_deviation', type = lambda x: bool(strtobool(x)), default = False, help = 'enable deviation')
    #parser.add_argument("--seed", type = int, default = 3380, help = "seed of the experiment")
    
    # plot arguments
    parser.add_argument('--plots_dir', type = str, default = 'plots', help = 'folder dir to save plot results')
    parser.add_argument('--exp_name', type = str, default = 'experiment', help = 'name of the experiment')
    parser.add_argument('--window_size', type = int, default = 100, help = 'window size of moving average')
    parser.add_argument('--plot_loss', type = lambda x: bool(strtobool(x)), default = False, help = 'enable plot loss')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args