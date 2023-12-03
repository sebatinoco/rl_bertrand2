import numpy as np
import gymnasium as gym
from scipy.optimize import minimize, fsolve
import torch

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

class Scaler:
    def __init__(self, moving_dim, dim):
        self.history = np.zeros((moving_dim, dim))
        self.moving_dim = moving_dim
        self.idx = 0
        self.count = 0
    
    def store(self, obs):
        
        self.history[self.idx] = obs
        self.idx += 1
        self.count += 1
        
        if self.idx >= self.moving_dim:
            self.idx = 0
        
    def transform(self, obs):
        
        self.store(obs)
        
        mean = np.mean(self.history, axis = 0) if self.count > self.moving_dim else np.mean(self.history[self.history != 0])
        std = np.std(self.history, axis = 0) if self.count > self.moving_dim else np.std(self.history[self.history != 0])
        
        return (obs - mean) / (std + 1e-8)

class BertrandEnv():
    def __init__(self, N, k, rho, timesteps, mu = 0.25, a_0 = 0, A = 2, c = 1, v = 3, xi = 0.2, 
                 inflation_start = 0, use_moving_avg = True, moving_dim = 10000, max_var = 2.0, 
                 dim_actions = 1):
        
        self.N = N # number of agents
        self.k = k # past periods to observe
        self.rho = rho # probability of changing prices
        self.a_0 = a_0 # base vertical differentiation index
        self.mu = mu # horizontal differentiation index
        self.A = np.array([A] * N) # vertical differentiation index
        self.c = c # marginal cost
        self.v = v # length of past inflations to predict current inflation
        self.xi = xi # price limit deflactor
        self.timesteps = timesteps
        self.inflation_start = inflation_start
        self.use_moving_avg = use_moving_avg
        self.moving_dim = moving_dim
        self.max_var = max_var
        self.trigger_deviation = False # trigger deviation
        self.altruist = False # altruist actions
        self.dim_actions = dim_actions
        self.pN = 1.0
        self.pM = 1.5
        self.rewards_scaler = Scaler(self.moving_dim, dim = self.N)
        
        assert v >= k, 'v must be greater or equal than k'
        
        self.inflation_model = torch.jit.load('inflation/inflation_model.pt')
        self.inflation_model.eval()
                
    def get_nash(self):
        def nash(p):

            '''
            Nash problem. Containes the derivatives of each agent with respect to its price.
            '''
            
            assert len(self.A_t) == len(p), "a must be equal size to p"

            sum_denominator = np.exp(self.a_0 / self.mu)
            for i in range(len(p)):
                sum_denominator += np.exp((self.A_t[i] - p[i]) / self.mu)

            result = []
            for i in range(len(p)):
                first_term = np.exp((self.A_t[i] - p[i]) / self.mu) / sum_denominator
                second_term = (np.exp((self.A_t[i] - p[i]) / self.mu) * (p[i] - self.c_t)) / (self.mu * sum_denominator)
                third_term = (p[i] - self.c_t) / self.mu

                fn = first_term * (1 + second_term - third_term)
                result.append(fn)

            return result
        
        #nash_solution = fsolve(nash, x0 = [1.0] * self.N)
        nash_solution = fsolve(nash, x0 = [self.pN] * self.N)
        
        assert all(round(price, 4) == round(nash_solution[0], 4) for price in nash_solution), \
        f"Nash price should be unique: {nash_solution}" # all prices are the same
        
        pN = nash_solution[0] # float
        
        return pN
    
    def get_monopoly(self):
        
        def monopoly(p):
            return -(p - self.c_t) * self.demand(p, self.A_t)
        
        #def objective(trial):
        #    solution = minimize(monopoly, x0 = trial.suggest_float('x0', 0, 10, step = 0.05)).x
        #    return monopoly(solution)

        #direction = 'minimize'
        #study = optuna.create_study(direction = direction, sampler = TPESampler())

        #study.optimize(objective, n_trials = 20)

        #x0 = study.best_trial.params['x0']
        
        #pM = minimize(monopoly, x0 = x0).x[0] # float
        pM = minimize(monopoly, x0 = self.pM).x[0] # float

        return pM
    
    def step(self, actions):
        
        '''
        Computes a step over the environment. Receives an action (array of prices) and return a tuple of (observation, reward, done, _)
        action: array of prices (np.array)
        '''
        
        self.action_history += [actions]
        
        # scale actions
        actions = [self.rescale_action(action) for action in actions]   
        
        if self.trigger_deviation:
            actions[0] = self.pN
            
        if self.altruist:
            actions[0] = self.pN
        
        # compute quantities
        quantities = self.demand(actions, self.A_t)
        self.quantities_history.append(quantities)
        # intrinsic reward: (p - c) * q
        rewards = [(actions[agent] - self.c_t) * quantities[agent] for agent in range(self.N)]
        #self.rewards_history.append(rewards)
        
        # update price history
        self.prices_history.append(actions)
        
        # obtain inflation
        inflation = self.get_inflation()
        self.inflation_history.append(inflation)
        
        # adjust moving avg -- sacar
        actions = np.array(actions, ndmin = 2)
        self.scaled_history = np.concatenate((self.scaled_history, actions), axis = 0)
        new_mean = np.mean(self.scaled_history[self.idx+1:self.idx+self.moving_dim+1, :], axis = 0)
        self.moving_avg = np.maximum(new_mean, 0) # update and moving avg always >= 0
        self.moving_history += [self.moving_avg]
        self.idx += 1
        
        # gather observation
        inflation = np.array(inflation, ndmin = 2, dtype = 'float32')
        cost = np.array(self.c_t, ndmin = 2, dtype = 'float32')
        past_prices = np.array(self.prices_history[-self.k:], dtype = 'float32')
        past_inflation = np.array(self.inflation_history[-self.k:], ndmin = 2, dtype = 'float32').T
        past_costs = np.array(self.costs_history[-self.k:], ndmin = 2, dtype = 'float32').T
        #ob_t1 = (inflation, cost, past_prices, past_inflation, past_costs)
        
        past_prices = past_prices - past_costs
        
        ob_t1 = (cost, past_prices, past_costs)
        ob_t1 = np.concatenate([element.flatten() for element in ob_t1])
        ob_t1 = self.obs_scaler.transform(ob_t1)
        
        self.timestep += 1
        if self.timestep > self.inflation_start:
            self.gen_inflation = True
         
        done = False if self.timestep < self.timesteps else True
        info = self.get_metric(rewards)
        
        #rewards = self.rewards_scaler.transform(rewards)
        #rewards = [(reward - self.pi_N) / (self.pi_M - self.pi_N) for reward in rewards] # rescale accordingly
        self.rewards_history.append(rewards)
        
        return ob_t1, rewards, done, info
    
    def init_history(self):
        self.inflation_history = [] # inflation history
        self.prices_history = [] # prices history
        self.quantities_history = [] # quantities history
        self.costs_history = [] # costs history
        self.nash_history = [] # nash prices history
        self.monopoly_history = [] # monopoly prices history
        self.rewards_history = [] # intrinsic rewards history
        self.metric_history = [] # collussion metric history
        self.pi_N_history = [] # nash utilities history
        self.pi_M_history = [] # monopoly utilities history
        self.moving_history = [] # moving avg history
        self.action_history = [] # action history
        self.A_history = [] # vertical diff history
    
    def init_boundaries(self):
        # set action boundaries
        #self.pN = self.c_t # get nash price
        #self.pM = (self.A_t + self.c_t) / 2 # get monopoly price
        
        self.pN = self.get_nash() # get nash price
        self.pM = self.get_monopoly() # get monopoly price
        
        self.pi_N = (self.pN - self.c) * self.demand([self.pN], self.A_t)[0]
        self.pi_M = (self.pM - self.c) * self.demand([self.pM], self.A_t)[0]
        
        self.pi_N_history += [self.pi_N]
        self.pi_M_history += [self.pi_M]

        assert self.pi_M > self.pi_N, f'monopoly profits should be higher than nash profits: {self.pi_N} vs {self.pi_M}'
        
        self.price_high = self.pM * (1 + self.xi)
        self.price_low = self.pN * (1 - self.xi)
        
        # limit prices
        expected_shocks = int((self.timesteps - self.inflation_start) * self.rho)
        expected_shocks = np.max([0, expected_shocks])
        #print('\n' + 'Expected shocks:', expected_shocks)
        
        self.expected_shocks = expected_shocks
        self.shocks = 0
        
        if self.use_moving_avg: # estoy usando esta via!!
            #self.action_range = [-self.max_var, self.max_var]
            self.action_range = [-0.5, self.max_var]
            #self.action_range = [self.pN, self.pM]
        else:
            #self.action_range = [self.price_low, self.price_high]
            self.action_range = [-3, 3]
    
    def reset(self):
        
        '''
        Resets the environment.
        '''
        
        # reset parameters
        self.gen_inflation = False
        self.A_t = self.A
        self.c_t = self.c
        self.timestep = 0
        
        # init history lists
        self.init_history()
        
        # init boundaries
        self.init_boundaries()
        
        # first observation
        self.prices_space = gym.spaces.Box(low = self.price_low, high = self.price_high, shape = (self.k, self.N), dtype = float) # prices space
        self.inflation_space = gym.spaces.Box(low = 0.015, high = 0.035, shape = (self.v,), dtype = float) # inflation space
        
        self.prices_history = [list(prices) for prices in self.prices_space.sample()] # init prices
        self.inflation_history = list(self.inflation_space.sample()) # init inflation
        
        self.scaled_history = np.random.uniform(self.price_low, self.price_high, (self.moving_dim, self.N))
        self.moving_avg = np.mean(self.scaled_history, axis = 0) # [moving_avg1, moving_avg2, ...]
        self.idx = 0
        
        self.costs_history = [self.c_t]
        for inflation in self.inflation_history[::-1]:
            self.costs_history = [self.costs_history[0] / (1 + inflation)] + self.costs_history
        
        ob_t = (
            #np.array(inflation, ndmin = 2, dtype = 'float32'), 
            np.array(self.c_t, ndmin = 2, dtype = 'float32'), 
            np.array(self.prices_history, ndmin = 2, dtype = 'float32') - self.c_t, 
            #np.array(self.inflation_history[-self.k:], ndmin = 2, dtype = 'float32').T, 
            np.array(self.costs_history[-self.k:], ndmin = 2, dtype = 'float32').T
            )
        
        ob_t = np.concatenate([dim.flatten() for dim in ob_t])
        self.obs_scaler = Scaler(moving_dim = self.moving_dim, dim = ob_t.shape[0])
        ob_t = self.obs_scaler.transform(ob_t)
        
        return ob_t
    
    def demand(self, prices, A):

        '''
        Returns the sold quantity in function of the vertical and horizontal differentiation, given the prices set.
        prices: Array of prices offered by agents (np.array)
        '''

        denominator = np.sum(np.exp((A - prices) / self.mu)) + np.exp(self.a_0 / self.mu)

        quantities = [np.exp((A[agent] - prices[agent]) / self.mu) / denominator for agent in range(len(prices))]

        return quantities
    
    def get_inflation(self):
        sample = np.random.rand()
        
        inflation_t = 0
        if (sample < self.rho) & (self.gen_inflation):
            
            with torch.no_grad():
                inflation_values = np.array(self.inflation_history) # transform to array
                inflation_values = inflation_values[inflation_values != 0][-self.v]
                inflation_values = torch.tensor(inflation_values).reshape(1, -1, 1).float()
                inflation_t = float(self.inflation_model(inflation_values).squeeze())
            
            dc = self.c_t * (inflation_t)
            
            self.c_t += dc # adjust marginal cost
            #self.a_t += dc # dc = da
            self.A_t = np.array([A + dc for A in self.A_t]) # dc = da
            
            #print('Calculating new equilibria...')
            self.pN = self.get_nash() # get nash price
            self.pM = self.get_monopoly() # get monopoly price
            
            self.pi_N = (self.pN - self.c_t) * self.demand([self.pN], self.A_t)[0]
            self.pi_M = (self.pM - self.c_t) * self.demand([self.pM], self.A_t)[0]
            #assert self.pi_M > self.pi_N, f'monopoly profits should be higher than nash profits: {self.pi_N} vs {self.pi_M}'
            
            self.price_high = self.pM * (1 + self.xi)
            self.price_low = self.pN * (1 - self.xi)
            
        self.costs_history += [self.c_t]
        self.A_history += [self.A_t[0]]
        self.nash_history += [self.pN]
        self.monopoly_history += [self.pM]
        self.pi_N_history += [self.pi_N]
        self.pi_M_history += [self.pi_M]
            
        return inflation_t
    
    def rescale_action(self, action):
        
        if self.dim_actions > 1:
            # dict between 0.5 and delta
            self.prices_dict = np.linspace(self.action_range[0], self.action_range[1], self.dim_actions)
            
            scaled_action = self.prices_dict[action] * self.c_t
        else:
            action = action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0 # scale variations
            
            if self.use_moving_avg:
                scaled_action = action * self.c_t # variations over cost
            else:
                scaled_action = action
        
        return np.max([scaled_action + self.c_t, 0.0])
    
    def get_metric(self, rewards, window = 1000):
        
        metric = (np.mean(rewards) - self.pi_N) / (self.pi_M - self.pi_N)
        self.metric_history.append(metric)
        
        return np.mean(self.metric_history[-window:])