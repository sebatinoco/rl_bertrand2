import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256, num_layers = 2, dropout = 0.1, random_state = 3380):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.random_state = random_state
        
        torch.manual_seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)

class DQNAgent():
    def __init__(self, dim_states, dim_actions, lr = 1e-3, gamma = 0.95, target_steps = 200, 
                 hidden_size = 256, epsilon = 0.9, beta = 5e-5, random_state = 3380, device = "cuda:0"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.dim_actions = dim_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta
        self.target_steps = target_steps
        self.target_count = 0
        self.t = 0
        self.random_state = random_state
        
        self.epsilon_history = []
        
        # instantiate networks
        self.network = DQN(dim_states, dim_actions, hidden_size, random_state=random_state).to(self.device)
        self.target_network = DQN(dim_states, dim_actions, hidden_size, random_state=random_state).to(self.device)
        
        self.target_network.load_state_dict(self.network.state_dict()) # load target network params
        
        self.optimizer = Adam(self.network.parameters(), lr = lr) # optimizer
        
    def select_action(self, state, greedy = False):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if np.random.random() > np.exp(-self.beta * self.t) or greedy:
            with torch.no_grad():
                action = torch.argmax(self.network(state), dim = 1).item()
        else:
            action = np.random.randint(0, self.dim_actions)
            
        self.epsilon_history += [np.exp(-self.beta * self.t)]
            
        self.t += 1
                
        return action
    
    def update(self, state, action, reward, state_t1, done):
        
        state = torch.tensor(state).to(self.device)
        action = torch.tensor(action, dtype = int).unsqueeze(1).to(self.device)
        reward = torch.tensor(reward).unsqueeze(1).to(self.device)
        state_t1 = torch.tensor(state_t1).to(self.device)
        done = torch.tensor(done).unsqueeze(dim = 1).to(self.device)
        
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