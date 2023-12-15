import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

def split_state(state, N, k, agent_idx):
    
    prices_costs = state[:, 1:-k].reshape(-1, k, N) # take just prices - costs
    self_price = prices_costs[:, :, agent_idx] # gather own series
    other_prices = np.delete(prices_costs, agent_idx, axis = 2).reshape(-1, k * (N - 1)) # gather rest of series
    cost_t = np.expand_dims(state[:, 0], 1)
    past_costs = state[:, -k:]
    
    return (self_price, other_prices, cost_t, past_costs)

class DQN(nn.Module):
    def __init__(self, N, k, output_size, hidden_size = 256, num_layers = 2, dropout = 0.1, random_state = 3380):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.random_state = random_state
        
        torch.manual_seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        
        self.self_price_embedding = nn.Linear(k, 1) 
        self.other_prices_embedding = nn.Linear((N-1) * k, 1)
        self.cost_t_embedding = nn.Linear(1, 1)
        self.past_costs_embedding = nn.Linear(k, 1)
        
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, self_price, other_prices, cost_t, past_costs):
        
        self_price = self.self_price_embedding(self_price)
        other_prices = self.other_prices_embedding(other_prices)
        cost_t = self.cost_t_embedding(cost_t)
        past_costs = self.past_costs_embedding(past_costs)
        
        state = torch.cat([self_price, other_prices, cost_t, past_costs], axis = 1)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)

class DQNAgent():
    def __init__(self, N, k, dim_actions, agent_idx, lr = 1e-3, gamma = 0.95, target_steps = 200, 
                 hidden_size = 256, epsilon = 0.9, beta = 5e-5, random_state = 3380):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dim_actions = dim_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta
        self.target_steps = target_steps
        self.target_count = 0
        self.t = 0
        self.random_state = random_state
        
        self.N = N
        self.k = k
        self.agent_idx = agent_idx
        
        # instantiate networks
        self.network = DQN(N, k, dim_actions, hidden_size).to(self.device)
        self.target_network = DQN(N, k, dim_actions, hidden_size).to(self.device)
        
        self.target_network.load_state_dict(self.network.state_dict()) # load target network params
        
        self.optimizer = Adam(self.network.parameters(), lr = lr) # optimizer
        
    def select_action(self, state, greedy = False):

        state = np.expand_dims(state, axis = 0)
        self_price, other_prices, cost_t, past_costs = split_state(state, self.N, self.k, self.agent_idx)
        
        self_price = torch.FloatTensor(self_price).to(self.device)
        other_prices = torch.FloatTensor(other_prices).to(self.device)
        cost_t = torch.FloatTensor(cost_t).to(self.device)
        past_costs = torch.FloatTensor(past_costs).to(self.device)
        state = (self_price, other_prices, cost_t, past_costs)

        if np.random.random() > np.exp(-self.beta * self.t):
            with torch.no_grad():
                action = torch.argmax(self.network(*state), dim = 1).item()
        else:
            action = np.random.randint(0, self.dim_actions)
            
        self.t += 1
                
        return action
    
    def update(self, state, action, reward, state_t1, done):
        
        self_price, other_prices, cost_t, past_costs = split_state(state, self.N, self.k, self.agent_idx)
        self_price = torch.FloatTensor(self_price).to(self.device)
        other_prices = torch.FloatTensor(other_prices).to(self.device)
        cost_t = torch.FloatTensor(cost_t).to(self.device)
        past_costs = torch.FloatTensor(past_costs).to(self.device)
        state = (self_price, other_prices, cost_t, past_costs)
        
        self_price_t1, other_prices_t1, cost_t_t1, past_costs_t1 = split_state(state_t1, self.N, self.k, self.agent_idx)
        self_price_t1 = torch.FloatTensor(self_price_t1).to(self.device)
        other_prices_t1 = torch.FloatTensor(other_prices_t1).to(self.device)
        cost_t_t1 = torch.FloatTensor(cost_t_t1).to(self.device)
        past_costs_t1 = torch.FloatTensor(past_costs_t1).to(self.device)
        state_t1 = (self_price_t1, other_prices_t1, cost_t_t1, past_costs_t1)
        
        action = torch.tensor(action, dtype = int).unsqueeze(1).to(self.device)
        reward = torch.tensor(reward).unsqueeze(1).to(self.device)
        done = torch.tensor(done).unsqueeze(dim = 1).to(self.device)
        
        self.target_count += 1
        self.update_network(state, action, reward, state_t1, done)
        if (self.target_count % self.target_steps) == 0:
            for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                target_param.data.copy_(param.data)
        
    def update_network(self, state, action, reward, state_t1, done):
        
        self_price, other_prices, cost_t, past_costs = state
        self_price_t1, other_prices_t1, cost_t_t1, past_costs_t1 = state_t1
        
        with torch.no_grad():
            target_max = torch.max(self.target_network(self_price_t1, other_prices_t1, cost_t_t1, past_costs_t1), dim = 1).values # max of Q values on t1
            td_target = reward.squeeze() + self.gamma * target_max * (1 - done.squeeze()) # fix the target
        
        old_val = self.network(self_price, other_prices, cost_t, past_costs).gather(1, action).squeeze() # prediction of network
        
        Q_loss = F.mse_loss(td_target, old_val)
        
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()