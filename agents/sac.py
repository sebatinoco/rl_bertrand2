import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import optim
import numpy as np

'''
SAC: observa inflacion, costos, costos pasados, inflacion pasada, precios pasados, media movil
'''

class SoftQNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi

class SACAgent:
  
    def __init__(self, dim_states, dim_actions, moving_dim = 10_000, max_var = 0.2, hidden_size = 256, 
                 gamma = 0.99, tau = 0.01, alpha = 0.2, Q_lr = 3e-4, actor_lr = 3e-4, alpha_lr = 3e-4, clip = 5,
                 beta = 2e-5, use_epsilon_greedy = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.obs_dim = dim_states
        self.action_dim = dim_actions
        
        self.mean_history = []
        self.std_history = []
        self.alpha_history = []
        self.action_history = []

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2
        self.moving_dim = moving_dim
        self.action_range = [-max_var, max_var]
        self.clip = clip
        self.beta = beta
        self.use_epsilon_greedy = use_epsilon_greedy
        
        # initialize networks 
        self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim, hidden_size).to(self.device)
        self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim, hidden_size).to(self.device)
        self.target_q_net1 = SoftQNetwork(self.obs_dim, self.action_dim, hidden_size).to(self.device)
        self.target_q_net2 = SoftQNetwork(self.obs_dim, self.action_dim, hidden_size).to(self.device)
        self.policy_net = PolicyNetwork(self.obs_dim, self.action_dim, hidden_size).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers 
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=Q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=Q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=actor_lr)

        # entropy temperature
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(dim_actions).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, state):
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()
        
        self.mean_history += [mean.item()]
        self.std_history += [std.item()]
        self.alpha_history += [self.alpha.item() if type(self.alpha) == torch.Tensor else self.alpha]
        
        normal = Normal(mean, std)
        z = normal.sample()

        if self.use_epsilon_greedy:
            z = self.epsilon_greedy(z, mean)
            
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        
        return action.item()
   
    def update(self, states, actions, rewards, next_states, dones):

        # to torch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)
        
        next_actions, next_log_pi = self.policy_net.sample(next_states) # sample action with noise
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)     
           
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        #nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.clip)
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        #nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.clip)
        self.q2_optimizer.step()
        
        # delayed update for policy network and target q networks -- UPDATE ACTOR
        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            #nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
            self.policy_optimizer.step()
        
            # target networks
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1
        
    def epsilon_greedy(self, z, mean):
        
        random_number = np.random.rand()
        epsilon = np.exp(-self.beta * self.update_step)
        
        if random_number > epsilon:
            return mean
        else:
            return z