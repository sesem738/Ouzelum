import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
              
        self.l1 = layer_init(nn.Linear(self.state, 256))
        self.l2 = layer_init(nn.Linear(256, 256))
        self.l3 = layer_init(nn.Linear(256, action_dim), std=0.01)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_dim)))
    
    def forward(self, state):
        x = F.tanh(self.l1(state))
        x = F.tanh(self.l2(x))
        action_mean = self.l3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action_mean, probs.log_prob(action).sum(1), probs.entropy().sum(1)

class Value(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = layer_init(nn.Linear(self.state, 256))
        self.l2 = layer_init(nn.Linear(256, 256))
        self.l3 = layer_init(nn.Linear(256, 1), std=1.0)

    def forward(self, state):
        x = F.tanh(self.l1(state))
        x = F.tanh(self.l2(x))
        x = self.l3(x)
        return x

