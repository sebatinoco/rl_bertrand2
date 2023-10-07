import numpy as np
import gymnasium as gym
import torch

import numpy as np
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

class LinearBertrandEnv():
    def __init__(self, N, k, rho, timesteps, A = 3, e = 1, c = 1, v = 3, xi = 0.2, 
                 inflation_start = 0, use_moving_avg = True, moving_dim = 1000, max_var = 2.0):
        
        self.N = N # number of agents
        self.k = k # past periods to observe
        self.rho = rho # probability of changing prices
        self.c = c # marginal cost
        self.v = v # length of past inflations to predict current inflation
        self.xi = xi # price limit deflactor
        self.moving_dim = moving_dim
        self.use_moving_avg = use_moving_avg
        self.max_var = max_var # max variation (moving avg)
        self.timesteps = timesteps # total steps per episode
        self.inflation_start = inflation_start # steps to begin with inflation
        self.trigger_deviation = False # trigger deviation
        self.altruist = False # altruist actions
        
        assert v >= k, 'v must be greater or equal than k'
        
        self.A = A # highest disposition to pay
        self.e = e # elasticity of demmand
            
        self.inflation_model = torch.jit.load('inflation/inflation_model.pt')
        self.inflation_model.eval()
                
    def step(self, action):
        
        '''
        Computes a step over the environment. Receives an action (array of prices) and return a tuple of (observation, reward, done, _)
        action: array of prices (np.array)
        '''
        
        self.action_history += [action]
        
        # scale actions
        action = [self.rescale_action(action[idx], idx) for idx in range(len(action))]    
        
        if self.trigger_deviation:
            action[0] = self.pN
            
        if self.altruist:
            action[0] = self.pN
        
        # compute quantities
        quantities = self.demand(action, self.A_t)
        self.quantities_history.append(quantities)
        
        # intrinsic reward: (p - c) * q
        reward = [(action[agent] - self.c_t) * quantities[agent] for agent in range(self.N)]
        self.rewards_history.append(reward)
        
        # update price history
        self.prices_history.append(action)
        
        # obtain inflation
        inflation = self.get_inflation()
        self.inflation_history.append(inflation)
        
        action = np.array(action, ndmin = 2)
        self.scaled_history = np.concatenate((self.scaled_history, action), axis = 0)
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
        moving_avg = np.array(self.moving_avg, ndmin = 2, dtype = 'float32')
        
        past_prices = past_prices - past_costs
        
        #ob_t1 = (inflation, cost, past_prices, past_inflation, past_costs, moving_avg)
        ob_t1 = (cost, past_prices, past_costs)
        ob_t1 = np.concatenate([element.flatten() for element in ob_t1])
        ob_t1 = self.obs_scaler.transform(ob_t1)
         
        self.timestep += 1
        if self.timestep > self.inflation_start:
            self.gen_inflation = True
         
        done = False if self.timestep < self.timesteps else True
        info = self.get_metric(reward)
        
        return ob_t1, reward, done, info
    
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
        self.A_history = [] # disposition to pay history
        
    def init_boundaries(self):
        # set action boundaries
        self.pN = self.c_t # get nash price
        self.pM = (self.A_t + self.c_t) / 2 # get monopoly price
        
        #self.nash_history += [self.pN]
        #self.monopoly_history += [self.pM]
        
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
        
        if self.use_moving_avg:
            self.action_range = [-self.max_var, self.max_var]
        else:
            self.action_range = [self.price_low, self.price_high]
            #self.action_range[1] = self.price_high * (1.035 ** expected_shocks)
            self.action_range[1] = self.price_high + self.c * (1.02 ** expected_shocks) # dA = dC
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
        
        self.prices_history = [list(prices) for prices in self.prices_space.sample()] # init prices
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
            np.array(self.costs_history[-self.k:], ndmin = 2, dtype = 'float32').T,
            #np.array(self.moving_avg, ndmin = 2, dtype = 'float32'),
            )
        
        ob_t = np.concatenate([dim.flatten() for dim in ob_t])
        
        self.obs_scaler = Scaler(moving_dim = self.moving_dim, dim = ob_t.shape[0])
        
        ob_t = self.obs_scaler.transform(ob_t)
        
        return ob_t
    
    def demand(self, prices, A):

        '''
        Returns the sold quantity in function of the prices set.
        prices: Array of prices offered by agents (np.array)
        '''

        p_min = np.min(prices)
        q_min = self.A_t - p_min * self.e

        quantities = [q_min if p == p_min and p < A else 0 for p in prices]
        
        #eq_count = np.count_nonzero(prices == p_min) # count p_min ocurrences
        #quantities = [q / eq_count for q in quantities]

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
            self.A_t += dc # dc = dA
            
            #print('Calculating new equilibria...')
            self.pN = self.c_t # get nash price
            self.pM = (self.A_t + self.c_t) / 2 # get monopoly price
            
            self.pi_N = (self.pN - self.c_t) * self.demand([self.pN], self.A_t)[0]
            self.pi_M = (self.pM - self.c_t) * self.demand([self.pM], self.A_t)[0]
            assert self.pi_M > self.pi_N, "monopoly profits should be higher than nash profits"
            
            self.price_high = self.pM * (1 + self.xi)
            self.price_low = self.pN * (1 - self.xi)
            
            #if self.use_moving_avg:
            #    self.action_range = [self.price_low, self.price_high]
            
            self.shocks += 1
        
        self.costs_history += [self.c_t]
        self.A_history += [self.A_t]
        self.nash_history += [self.pN]
        self.monopoly_history += [self.pM]
        self.pi_N_history += [self.pi_N]
        self.pi_M_history += [self.pi_M]
            
        return inflation_t
    
    def rescale_action(self, action, agent_idx):
        
        action = action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0 # scale variations
        
        if self.use_moving_avg:
            #scaled_action = np.max([self.moving_avg[agent_idx] * (1 + action), 0.0]) # scale action
            scaled_action = np.max([action * self.c_t, 0.0]) # variations over cost
        else:
            scaled_action = action
        
        return scaled_action + self.c_t
    
    def get_metric(self, rewards, window = 1000):
        
        metric = (np.max(rewards) - self.pi_N) / (self.pi_M - self.pi_N)
        self.metric_history.append(metric)
        
        return np.mean(self.metric_history[-window:])