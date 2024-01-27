import numpy as np

class ReplayBuffer():
    def __init__(self, dim_states, N, buffer_size, sample_size):
        
        self.buffer_st = np.zeros((buffer_size, dim_states), dtype = 'float32')
        self.buffer_rt = np.zeros((buffer_size, N), dtype = 'float32')
        self.buffer_at = np.zeros((buffer_size, N), dtype = 'float32')
        self.buffer_st1 = np.zeros((buffer_size, dim_states), dtype = 'float32')
        self.buffer_done = np.zeros(buffer_size, dtype = 'float32')
        
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
            
        #if self.exps_stored < self.sample_size:
        #    raise ValueError('Not enough samples stored on buffer')
        #elif self.sample_size < self.exps_stored < self.buffer_size:
        #    sample_idxs = np.random.randint(0, self.exps_stored, size = self.sample_size)
        #else:
        #    sample_idxs = np.random.randint(0, self.buffer_size, size = self.sample_size)
        
        return (self.buffer_st[sample_idxs], # states
                self.buffer_at[sample_idxs][:, idx], # actions
                self.buffer_rt[sample_idxs][:, idx], # rewards
                self.buffer_st1[sample_idxs], # states_t1
                self.buffer_done[sample_idxs], # done
                )