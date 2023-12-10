import numpy as np

def split_state(state, N, k, idx):
    
    prices_costs = state[:, 1:-k].reshape(-1, k, N) # take just prices - costs
    self_price = prices_costs[:, :, idx] # gather own series
    other_prices = np.delete(prices_costs, idx, axis = 2).reshape(-1, k * (N - 1)) # gather rest of series
    cost_t = np.expand_dims(state[:, 0], 1)
    past_costs = state[:, -k:]
    
    return (self_price, other_prices, cost_t, past_costs)

class ReplayBuffer():
    def __init__(self, dim_states, N, k, buffer_size, sample_size):
        
        self.buffer_st = np.zeros((buffer_size, dim_states), dtype = 'float32')
        self.buffer_rt = np.zeros((buffer_size, N), dtype = 'float32')
        self.buffer_at = np.zeros((buffer_size, N), dtype = 'float32')
        self.buffer_st1 = np.zeros((buffer_size, dim_states), dtype = 'float32')
        self.buffer_done = np.zeros(buffer_size, dtype = 'float32')
        
        self.N = N
        self.k = k
        
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        
        self.idx = 0
        self.exps_stored = 0
    
    def store_transition(self, ob_t, a_t, r_t, ob_t1, done_t):

        # store transition
        self.buffer_st[self.idx] = ob_t
        self.buffer_at[self.idx] = a_t
        self.buffer_rt[self.idx] = r_t
        self.buffer_st1[self.idx] = ob_t1
        self.buffer_done[self.idx] = done_t
        
        self.exps_stored += 1
    
        # update idx
        if self.idx == self.buffer_size - 1:
            # reset count
            self.idx = 0
        else:
            # increment count
            self.idx += 1    
    
    def sample(self, idx):
        
        if self.exps_stored < self.sample_size:
            raise ValueError('Not enough samples stored on buffer')
        
        if self.sample_size <= self.exps_stored < self.buffer_size:
            sample_idxs = np.random.randint(0, self.exps_stored, size = self.sample_size)
        else:
            sample_idxs = np.random.randint(0, self.sample_size, size = self.sample_size)
        
        return (split_state(self.buffer_st[sample_idxs]), # states
                self.buffer_at[sample_idxs][:, idx], # actions
                self.buffer_rt[sample_idxs][:, idx], # rewards
                split_state(self.buffer_st1[sample_idxs]), # states_t1
                self.buffer_done[sample_idxs], # done
                )