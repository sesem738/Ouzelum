import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self,single_observation_space,single_action_space):
        super().__init__()
        self.rpo_alpha = 0.5
        self.obs_space = single_observation_space
        self.act_space = single_action_space
        
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(self.obs_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(256, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor_mean = layer_init(nn.Linear(128, np.prod(self.act_space.shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.act_space.shape)))

    def get_states(self, state, lstm_state, done):
        hidden = self.network(state)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    
    def forward(self, state, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(state, lstm_state, done)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else: # new to RPO
            # sample again to add stochasticity, for the policy update
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to("cuda:0")
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), lstm_state

class Critic(nn.Module):
    def __init__(self,single_observation_space) -> None:
        super().__init__()
        self.obs_space = single_observation_space
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(self.obs_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
    
    def forward(self,state):
        x = self.critic(state)
        return x
