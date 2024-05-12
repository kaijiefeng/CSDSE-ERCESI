import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


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


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()

class SquashedGaussianMLPActor_DSE(nn.Module):

    def __init__(self, obs_dim, act_dim_list, act_limit_list):
        super().__init__()

        self.input = torch.nn.Linear(obs_dim, 128)
        self.hidden = torch.nn.Linear(128, 64)

        self.mu_output_list = list()
        for act_dim in act_dim_list:
            self.mu_output_list.append(torch.nn.Linear(64, act_dim))
        self.mu_output = torch.nn.ModuleList(self.mu_output_list)
        
        self.log_std_output_list = list()
        for act_dim in act_dim_list:
            self.log_std_output_list.append(torch.nn.Linear(64, act_dim))
        self.log_std_output = torch.nn.ModuleList(self.log_std_output_list)

        self.act_limit_list = act_limit_list

    def forward(self, obs, act_idx, deterministic=False, with_logprob=True):
        if(not torch.is_tensor(act_idx)):
            out1 = F.relu(self.input(obs))
            out2 = F.relu(self.hidden(out1))
            mu = self.mu_output[act_idx](out2)
            log_std = self.log_std_output[act_idx](out2)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)

            # Pre-squash distribution and sample
            pi_distribution = Normal(mu, std)
            if deterministic:
                # Only used for evaluating policy at test time.
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()

            if with_logprob:
                # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
                # NOTE: The correction formula is a little bit magic. To get an understanding 
                # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
                # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
                # Try deriving it yourself as a (very difficult) exercise. :)
                logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
                logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
            else:
                logp_pi = None

            pi_action = torch.tanh(pi_action)
            pi_action = pi_action #* self.act_limit_list[act_idx]

            return pi_action, logp_pi
        else:
            act_idx = act_idx.numpy()
            pi_action_list = list()
            logp_pi_t = None

            for obs_i, act_idx_i in zip(obs, act_idx):
                out1 = F.relu(self.input(obs_i))
                out2 = F.relu(self.hidden(out1))
                mu = self.mu_output[int(act_idx_i)](out2)
                log_std = self.log_std_output[int(act_idx_i)](out2)
                log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
                std = torch.exp(log_std)    

                # Pre-squash distribution and sample
                pi_distribution = Normal(mu, std)
                if deterministic:
                    # Only used for evaluating policy at test time.
                    pi_action = mu
                else:
                    pi_action = pi_distribution.rsample()    

                if with_logprob:
                    # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
                    # NOTE: The correction formula is a little bit magic. To get an understanding 
                    # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
                    # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
                    # Try deriving it yourself as a (very difficult) exercise. :)
                    logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
                    logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
                else:
                    logp_pi = None    

                pi_action = torch.tanh(pi_action)
                pi_action = pi_action #* self.act_limit_list[act_idx] 

                pi_action_list.append(pi_action)
                if(logp_pi_t == None): logp_pi_t = logp_pi.view(-1)
                else: logp_pi_t = torch.cat([logp_pi_t, logp_pi.view(-1)], dim = -1)
            return pi_action_list, logp_pi_t   

class MLPQFunction_DSE(nn.Module):

    def __init__(self, obs_dim, act_dim_list):
        super().__init__()

        self.input_list = list()
        for act_dim in act_dim_list:
            self.input_list.append(torch.nn.Linear(obs_dim + act_dim, 256))
        self.input = torch.nn.ModuleList(self.input_list)

        self.hidden = torch.nn.Linear(256, 128)
        self.output = torch.nn.Linear(128, 1)

    def forward(self, obs, act, act_idx):
        if(not isinstance(act, list)):
            out1 = F.relu(self.input[act_idx](torch.cat([obs, act], dim=-1)))
            out2 = F.relu(self.hidden(out1))
            q = self.output(out2)

            return torch.squeeze(q, -1)
        else:
            act_idx = act_idx.numpy()
            q = None
            for obs_i, act_i, act_idx_i in zip(obs, act, act_idx):
                out1 = F.relu(self.input[int(act_idx_i)](torch.cat([obs_i, act_i], dim=-1)))
                out2 = F.relu(self.hidden(out1))
                q_i = self.output(out2)

                if(q == None): q = q_i.view(1)
                else: q = torch.cat([q, q_i.view(1)], dim = -1)

            return torch.squeeze(q, -1)

class MLPActorCritic_DSE(nn.Module):

    def __init__(self, obs_dim, act_dim_list, act_limit_list):
        super().__init__()

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor_DSE(obs_dim, act_dim_list, act_limit_list)
        self.q1 = MLPQFunction_DSE(obs_dim, act_dim_list)
        self.q2 = MLPQFunction_DSE(obs_dim, act_dim_list)

    def act(self, obs, act_idx, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, act_idx, deterministic, False)
            return a.numpy()