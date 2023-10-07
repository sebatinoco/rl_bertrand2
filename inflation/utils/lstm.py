import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout = 0.0, random_state = 3380):
        super(LSTM, self).__init__()
        
        # Fijamos semilla 
        if random_state:
            torch.manual_seed(random_state)
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout = dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        print(self.fc)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        x, hidden = self.lstm(x, (h_0, c_0))
        
        # gather last state
        x = x[:, -1, :]
        
        # fully connected
        out = self.fc(x)
        
        return out