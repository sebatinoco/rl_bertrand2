import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256, num_layers = 2, dropout = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embedding = nn.Linear(1, hidden_size)
        # input: N x seq_large (k) x features (N)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers,
                            batch_first = True, dropout = dropout)
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, state):
        
        # unpack
        inflation, past_prices, past_inflation = state
        
        history = torch.cat([past_prices, past_inflation], dim = 2)
        
        # inflation embedding
        inflation = F.relu(self.embedding(inflation))
        
        # history lstm
        h_0 = Variable(torch.randn(
            self.num_layers, history.size(0), self.hidden_size))
        
        c_0 = Variable(torch.randn(
            self.num_layers, history.size(0), self.hidden_size))
        
        history, hidden = self.lstm(history, (h_0, c_0))
        history = F.relu(history[:, -1, :])
        
        # concatenate
        x = torch.cat([inflation, history], dim = 1)
        
        # output -1 to 1
        x = self.fc(x)
        
        return x

class DQNAgent():
    def __init__(self, N, lr = 1e-3, gamma = 0.99, target_steps = 200, hidden_size = 256, epsilon = 0.9, epsilon_decay = 0.99, dim_actions = 15):
        
        # N: number of agents (needed to generate network)
        
        self.dim_actions = dim_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_steps = target_steps
        self.target_count = 0
        
        # instantiate networks
        self.network = DQN(N + 1, dim_actions, hidden_size)
        self.target_network = DQN(N + 1, dim_actions, hidden_size)
        
        self.target_network.load_state_dict(self.network.state_dict()) # load target network params
        
        self.optimizer = Adam(self.network.parameters(), lr = lr) # optimizer
        
    def select_action(self, state, action_high, action_low):

        inflation, past_prices, past_inflation = state
        inflation = torch.tensor(inflation)
        past_prices = torch.tensor(past_prices).unsqueeze(0)
        past_inflation = torch.tensor(past_inflation).unsqueeze(0)
        
        state = (inflation, past_prices, past_inflation)

        if np.random.random() > self.epsilon:
            with torch.no_grad():
                action = torch.argmax(self.network(state), dim = 1).item()
        else:
                action = np.random.randint(0, self.dim_actions)
                self.epsilon *= self.epsilon_decay
                
        return action
    
    def update(self, state, action, reward, state_t1, done):
        
        inflation = torch.tensor(np.array([s[0] for s in state])).squeeze(2)
        past_prices = torch.tensor(np.array([s[1] for s in state]))
        past_inflation = torch.tensor(np.array([s[2] for s in state]))
        
        inflation_t1 = torch.tensor(np.array([s[0] for s in state_t1])).squeeze(2)
        past_prices_t1 = torch.tensor(np.array([s[1] for s in state_t1]))
        past_inflation_t1 = torch.tensor(np.array([s[2] for s in state_t1]))    
        
        state = (inflation, past_prices, past_inflation)
        action = torch.tensor(action, dtype = int).unsqueeze(1)
        reward = torch.tensor(reward).unsqueeze(1)
        state_t1 = (inflation_t1, past_prices_t1, past_inflation_t1)
        done = torch.tensor(done).unsqueeze(dim = 1)
        
        self.target_count += 1
        self.update_network(state, action, reward, state_t1, done)
        if (self.target_count % self.target_steps) == 0:
            for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                target_param.data.copy_(param.data)
        
    def update_network(self, state, action, reward, state_t1, done):
        
        with torch.no_grad():
            target_max = torch.max(self.target_network(state_t1), dim = 1).values # max of Q values on t1
            td_target = reward.squeeze() + self.gamma * target_max * (1 - done.squeeze()) # fix the target
        
        old_val = self.network(state).gather(1, action).squeeze() # prediction of network
        
        Q_loss = F.mse_loss(td_target, old_val)
        
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()