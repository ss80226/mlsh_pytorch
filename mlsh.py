import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


_init = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
    x, 0), np.sqrt(2))

class MLSHMasterPolicy(nn.Module):
    def __init__(self, state_dim, subpolicies_num):
        super(MLSHMasterPolicy, self).__init__()
        # self._A = action_dim
        self._mlp = nn.Sequential(_init(nn.Linear(state_dim, 64)),
                                  nn.ReLU(), _init(nn.Linear(64, 64)),
                                  nn.ReLu(), _init(nn.Linear(64, subpolicies_num)))
        return
    
    def forward(self, state):
        return self._mlp(state) # (N * k)

    def act(self, state):
        x = F.softmax(self.forward(state))
        d = dist.categorical.Categorical(probs=x)
        action = d.sample()
        log_prob, ent = d.log_prob(action), d.entropy()
        return action, log_prob
    
    def evaluate_action(self, state, action):
        '''
        Estimate the log probability of action and entropy of the 
        action distribution

        input:
        - state: state of the agent with shape (N, S)
        - action: action with shape (N, A) we intend to estimate 

        output:
        - log_prob: the log probability of the action in the distribution 
                    estimated from the policy
        - ent: the entropy of the action distribution
        '''
        
        x = F.softmax(self.forward(state))
        d = dist.categorical.Categorical(probs=x)
        log_prob, ent = d.log_prob(action), d.entropy()
        return log_prob, ent

class MasterValueNet(nn.Module):
    def __init__(self, state_dim):
        super(MasterValueNet, self).__init__()
        self._mlp = nn.Sequential(_init(nn.Linear(state_dim, 64)),
                                  nn.ReLU(), _init(nn.Linear(64, 64)),
                                  nn.ReLu(), _init(nn.Linear(64, 1)))
        return
    
    def forward(self, state):
        return self._mlp(state)

class MLSHSubPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLSHMasterPolicy, self).__init__()
        self._A = action_dim
        self._mlp = nn.Sequential(_init(nn.Linear(state_dim, 64)),
                                  nn.ReLU(), _init(nn.Linear(64, 64)),
                                  nn.ReLu(), _init(nn.Linear(64, self._A*2)))
        return
    
    def forward(self, state):
        mu_sigma = self._mlp(state) # (N * A*2)
        return torch.tanh(mu_sigma[:, :self._A]), F.softplus(mu_sigma[:, self._A:]) # mu, sigma

    def act(self, state):
        mu, sigma = self.forward(state)
        d = dist.normal.Normal(mu, sigma)
        action = d.sample()
        log_prob, ent = d.log_prob(action), d.entropy()

        return action, log_prob
    
    def evaluate_action(self, state, action):
        '''
        Estimate the log probability of action and entropy of the 
        action distribution

        input:
        - state: state of the agent with shape (N, S)
        - action: action with shape (N, A) we intend to estimate 

        output:
        - log_prob: the log probability of the action in the distribution 
                    estimated from the policy
        - ent: the entropy of the action distribution
        '''
        
        mu, sigma =self.forward(state)
        d = dist.normal.Normal(mu, sigma)
        log_prob, ent = d.log_prob(action), d.entropy()

        return action, log_prob

class SubValueNet(nn.Module):
    def __init__(self, state_dim):
        super(MasterValueNet, self).__init__()
        self._mlp = nn.Sequential(_init(nn.Linear(state_dim, 64)),
                                  nn.ReLU(), _init(nn.Linear(64, 64)),
                                  nn.ReLu(), _init(nn.Linear(64, 1)))
        return
    
    def forward(self, state):
        return self._mlp(state)