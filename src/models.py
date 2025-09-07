# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(layer: torch.nn.modules):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight.data, gain)
        if hasattr(layer.bias, 'data'): layer.bias.data.fill_(0.0)


def ln_activ(x: torch.Tensor, activ: Callable):
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ: str='elu'):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, output_dim)

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, x: torch.Tensor):
        y = ln_activ(self.l1(x), self.activ)
        y = ln_activ(self.l2(y), self.activ)
        return self.l3(y)


class Encoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, pixel_obs: bool,
        num_bins: int=65, zs_dim: int=512, za_dim: int=256, zsa_dim: int=512, hdim: int=512, activ: str='elu'):
        super().__init__()
        if pixel_obs:
            self.zs = self.cnn_zs
            self.zs_cnn1 = nn.Conv2d(state_dim, 32, 3, stride=2)
            self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
            self.zs_lin = nn.Linear(1568, zs_dim)
        else:
            self.zs = self.mlp_zs
            self.zs_mlp = BaseMLP(state_dim, zs_dim, hdim, activ)

        self.za = nn.Linear(action_dim, za_dim)
        self.zsa = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ)
        self.model = nn.Linear(zsa_dim, num_bins + zs_dim + 1)

        self.zs_dim = zs_dim

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], dim=-1))


    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        zsa = self.forward(zs, action)
        dzr = self.model(zsa)
        return dzr[:,0:1], dzr[:,1:self.zs_dim+1], dzr[:,self.zs_dim+1:] # done, zs, reward


    def cnn_zs(self, state: torch.Tensor):
        state = state/255. - 0.5
        zs = self.activ(self.zs_cnn1(state))
        zs = self.activ(self.zs_cnn2(zs))
        zs = self.activ(self.zs_cnn3(zs))
        zs = self.activ(self.zs_cnn4(zs)).reshape(state.shape[0], -1)
        return ln_activ(self.zs_lin(zs), self.activ)


    def mlp_zs(self, state: torch.Tensor):
        return ln_activ(self.zs_mlp(state), self.activ)


class Policy(nn.Module):
    def __init__(self, action_dim: int, discrete: bool, gumbel_tau: float=10, zs_dim: int=512, hdim: int=512, activ: str='relu'):
        super().__init__()
        self.policy = BaseMLP(zs_dim, action_dim, hdim, activ)
        self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if discrete else torch.tanh
        self.discrete = discrete


    def forward(self, zs: torch.Tensor):
        pre_activ = self.policy(zs)
        action = self.activ(pre_activ)
        return action, pre_activ


    def act(self, zs: torch.Tensor):
        action, _ = self.forward(zs)
        return action

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hdim: int=512, activ: str='elu'):
        super().__init__()
        self.q1 = BaseMLP(input_dim, hdim, hdim, activ)
        self.q2 = nn.Linear(hdim, 1)

        self.activ = getattr(F, activ)
        self.apply(weight_init)

    def forward(self, zsa: torch.Tensor):
        zsa = ln_activ(self.q1(zsa), self.activ)
        return self.q2(zsa).squeeze(-1)

class Value(nn.Module):
    def __init__(self, zsa_dim: int=512, hdim: int=512, activ: str='elu'):
        super().__init__()

        self.q1 = ValueNetwork(zsa_dim, hdim, activ)
        self.q2 = ValueNetwork(zsa_dim, hdim, activ)


    def forward(self, zsa: torch.Tensor):
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)

class NuNetwork(nn.Module):
    def __init__(self, input_dim: int, hdim: int=512, activ: str='elu'):
        super().__init__()
        self.net = BaseMLP(input_dim, 1, hdim, activ) # BaseMLP handles most layers

        # Custom initialization for the *final layer* only
        # This forces the initial output to be around -2.0
        final_layer = self.net.l3
        final_layer.weight.data.uniform_(-1e-2, 1e-2) # Almost zero weights
        final_layer.bias.data.fill_(-2.0) # Strong negative bias

    def forward(self, zsa: torch.Tensor):
        return self.net(zsa).squeeze(-1)

class Gumbel(nn.Module):
    def __init__(self, zsa_dim: int=512, zs_dim: int=512, hdim: int=512, activ: str='elu'):
        super().__init__()
        self.mu = ValueNetwork(zsa_dim, hdim, activ)
        self.nu = NuNetwork(zs_dim, hdim, activ)

        self.activ = getattr(F, activ)
        # self.apply(weight_init)

    def distribuition(self, zsa, zs):
        mu, nu = self.mu(zsa), self.nu(zs)

        return torch.distributions.gumbel.Gumbel(mu, nu.exp())
    

    def loss(self, r, mu_q, nu_q, mu_p, nu_p, not_done_rl, gamma=1):
        not_done_rl, r = not_done_rl.squeeze(-1), r.squeeze(-1)
    
        balance = not_done_rl.sum()/not_done_rl.numel()
        not_done_rl = not_done_rl.bool()

        if balance!=1:
            mu_done, nu_done, r_done= mu_q[~not_done_rl], nu_q[~not_done_rl], r[~not_done_rl]
            z = (mu_done - r_done) * torch.exp(-nu_done)
            loss_done = nu_done - z + torch.exp(z)

        mu_q, nu_q, mu_p, nu_p, r = mu_q[not_done_rl], nu_q[not_done_rl], mu_p[not_done_rl], nu_p[not_done_rl], r[not_done_rl]
        # This part for calculating target parameters is also correct.
        with torch.no_grad():
            logsumexp = torch.logsumexp(mu_p * torch.exp(-nu_p).unsqueeze(1), dim=1)
            mu_p_target = r + gamma * torch.exp(nu_p) * logsumexp
            nu_p_target = nu_p + np.log(gamma)

        # Re-assign for clarity in the loss formula
        print(
            f"μ pred max: {mu_p.max(dim=1).values.mean():>8.4f}  |  "
            f"μ target: {mu_p_target.mean():>8.4f}  |  "
            f"μ_q: {mu_q.mean():>8.4f}  ||  "
            f"ν pred: {nu_p.mean():>8.4f}  |  "
            f"ν target: {nu_p_target.mean():>8.4f}"
        )
        mu_p = mu_p_target
        nu_p = nu_p_target

        # Loss calculation based on the paper's formulas
        z = (mu_q - mu_p) * torch.exp(-nu_q)
        d = torch.exp(nu_p - nu_q)

        # CORRECTED LOSS: Note the multiplication (*) in the last term
        loss = nu_q - z + np.euler_gamma * d + torch.exp(z) * torch.exp(torch.lgamma(1 + d))
    
        # CORRECTED KL: Note the subtraction (-) to isolate the KL term
        KL = loss - (nu_p + np.euler_gamma + 1)
    

        print(f"loss: {loss.mean().item():.4f}, KL: {KL.mean().item():.4f}, KL_r!=0 : {KL[r!=0].mean().item():.4f}")
        if balance==1:
            return loss.mean(), KL
        
        out_KL = torch.empty(not_done_rl.numel(), device = r.device)
        out_KL[not_done_rl]=KL
        out_KL[~not_done_rl]=loss_done

        return loss.mean()*balance + (1-balance)*loss_done.mean(), out_KL