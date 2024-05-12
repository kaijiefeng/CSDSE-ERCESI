import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import pdb


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)



class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class MLPCategoricalActor_DSE(nn.Module):

    def __init__(self, obs_dim, act_dim_list):
        super(MLPCategoricalActor_DSE, self).__init__()

        self.input = torch.nn.Linear(obs_dim, 128)
        self.hidden = torch.nn.Linear(128, 64)
        self.output_list = list()
        for act_dim in act_dim_list:
            self.output_list.append(torch.nn.Linear(64, act_dim))
        self.output = torch.nn.ModuleList(self.output_list)

    def _distribution(self, obs, act_idx):
        if(not torch.is_tensor(act_idx)):
            out1 = torch.nn.functional.relu(self.input(obs))
            out2 = torch.nn.functional.relu(self.hidden(out1))
            logits = torch.tanh(self.output[act_idx](out2))
            return Categorical(logits=logits)
        else:
            act_idx = act_idx.numpy()
            pi_list = list()
            for obs_i, act_idx_i in zip(obs, act_idx):
                out1 = torch.nn.functional.relu(self.input(obs_i))
                out2 = torch.nn.functional.relu(self.hidden(out1))
                logits = torch.tanh(self.output[int(act_idx_i)](out2))
                pi_list.append(Categorical(logits=logits)) 
            return pi_list               


    def _log_prob_from_distribution(self, pi, act):
        if(not isinstance(pi, list)):
            return pi.log_prob(act)
        else:
            logp_a = None
            for pi_i, act_i in zip(pi, act):
                
                if(logp_a is None): logp_a = pi_i.log_prob(act_i).view(1)
                else: logp_a = torch.cat((logp_a, pi_i.log_prob(act_i).view(1)), -1)            
            return logp_a

    def forward(self, obs, act_idx, act= None):
        pi = self._distribution(obs, act_idx)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCritic_DSE(nn.Module):

    def __init__(self, obs_dim):
        super(MLPCritic_DSE, self).__init__()

        self.input = torch.nn.Linear(obs_dim, 256)
        self.hidden = torch.nn.Linear(256, 128)
        self.output = torch.nn.Linear(128,1)

    def forward(self, obs):
        out1 = torch.nn.functional.relu(self.input(obs))
        out2 = torch.nn.functional.relu(self.hidden(out1))
        return torch.squeeze(self.output(out2), -1)

class MLPActorCritic_DSE(nn.Module):

    def __init__(self, obs_dim, act_dim_list):
        super(MLPActorCritic_DSE, self).__init__()

        self.pi = MLPCategoricalActor_DSE(obs_dim, act_dim_list)
        self.v = MLPCritic_DSE(obs_dim)

    def step(self, obs, act_idx):
        with torch.no_grad():
            pi = self.pi._distribution(obs, act_idx)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class RNNCategoricalActor_DSE(nn.Module):

    def __init__(self, obs_dim, act_dim_list):
        super(RNNCategoricalActor_DSE, self).__init__()

        self.input_lenth = len(self.action_scale_list)
        self.rnn = torch.nn.LSTM(
            input_size = self.input_lenth+1,
            hidden_size = 128,
            num_layers = 1,
            batch_first = True
        )
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = list()
        for act_dim in self.act_dim_list:
            self.fc2.append(torch.nn.Linear(64, act_dim))
        self.fc2 = torch.nn.ModuleList(self.fc2)

    def _distribution(self, obs, act_idx, rnn_state):
        if(not torch.is_tensor(act_idx)):
            norm_act_idx = act_idx/self.lenth
            input = torch.cat((obs, torch.tensor(norm_act_idx).float().view(1)), dim = -1)
            out_rnn, rnn_state = self.rnn(input.view(1, 1, self.input_lenth+1), rnn_state)
            out2 = torch.nn.functional.relu(self.fc1(out_rnn[0,0,:]))
            out3 = self.fc2[act_idx](out2)
            logits = torch.tanh(out3)
            return Categorical(logits=logits), rnn_state
        else:
            act_idx = act_idx.numpy()
            pi_list = list()
            rnn_state_list = list()
            for obs_i, act_idx_i, rnn_state_i in zip(obs, act_idx, rnn_state):
                norm_act_idx = act_idx_i/self.lenth
                input = torch.cat((obs_i, torch.tensor(norm_act_idx).float().view(1)), dim = -1)
                out_rnn, rnn_state_i = self.rnn(input.view(1, 1, self.input_lenth+1), rnn_state_i)
                out2 = torch.nn.functional.relu(self.fc1(out_rnn[0,0,:]))
                out3 = self.fc2[act_idx_i](out2)
                logits = torch.tanh(out3)
                pi_list.append(Categorical(logits=logits))
                rnn_state_list.append(rnn_state_i)
            return pi_list, rnn_state_list

    def _log_prob_from_distribution(self, pi, act):
        if(not isinstance(pi, list)):
            return pi.log_prob(act)
        else:
            logp_a = None
            for pi_i, act_i in zip(pi, act):
                if(logp_a is None): logp_a = pi_i.log_prob(act_i).view(1)
                else: logp_a = torch.cat((logp_a, pi_i.log_prob(act_i).view(1)), -1)            
            return logp_a

    def forward(self, obs, act_idx, rnn_state, act= None):
        pi, rnn_state = self._distribution(obs, act_idx, rnn_state)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a, rnn_state

class RNNCritic_DSE(nn.Module):

    def __init__(self, obs_dim):
        super(RNNCritic_DSE, self).__init__()

        self.input = torch.nn.Linear(obs_dim, 256)
        self.hidden = torch.nn.Linear(256, 128)
        self.output = torch.nn.Linear(128,1)

        self.input_lenth = obs_dim
        self.rnn = torch.nn.LSTM(
            input_size = self.input_lenth+1,
            hidden_size = 256,
            num_layers = 1,
            batch_first = True
        )
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, obs, act_idx, rnn_state):
        norm_act_idx = act_idx/self.lenth
        input = torch.cat((obs, torch.tensor(norm_act_idx).float().view(1)), dim = -1)
        out_rnn, rnn_state = self.rnn(input.view(1, 1, self.input_lenth+1), rnn_state)
        out2 = torch.nn.functional.relu(self.fc1(out_rnn[0,0,:]))
        return torch.squeeze(self.output(out2), -1), rnn_state

class RNNActorCritic_DSE(nn.Module):

    def __init__(self, obs_dim, act_dim_list):
        super(RNNActorCritic_DSE, self).__init__()

        self.pi = RNNCategoricalActor_DSE(obs_dim, act_dim_list)
        self.v = RNNCritic_DSE(obs_dim)

    def step(self, obs, act_idx, rnn_state):
        with torch.no_grad():
            pi = self.pi._distribution(obs, act_idx, rnn_state)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs, act_idx, rnn_state)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
