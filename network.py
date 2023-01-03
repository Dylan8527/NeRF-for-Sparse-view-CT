'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-12-04 20:00:23
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-12-17 22:31:15
FilePath: \code\network.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp

class NeRFNetwork(nn.Module):
    def __init__(self,
        encoding='hashgrid',
        num_layers=2,
        hidden_dim=64,
        bound=1.,
        **kwargs):
        super().__init__()

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.register_buffer('bound', torch.tensor(bound))

        self.encoder, self.in_dim = get_encoder(encoding, 
                                    input_dim=2, 
                                    num_levels=16, 
                                    level_dim=4, 
                                    base_resolution=2, 
                                    log2_hashmap_size=19, 
                                    desired_resolution=2048 * bound)

        in_dims = [self.in_dim] + [self.hidden_dim] * (self.num_layers - 1)
        out_dims = [self.hidden_dim] * (self.num_layers - 1) + [1]

        sigma_net = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)]
        self.sigma_net = nn.ModuleList(sigma_net)

        self.act = nn.Softplus(10)

    def forward(self, x: torch.tensor, perturb=False)->torch.tensor:
        # x.size() = [B, N, T, 2]
        B, N, T = x.shape[:3]
        x = x.view(-1, T, 2) # view as [N, T, 2]
        
        device = x.device
        rays_o = x[:, 0] # [N, 2]
        rays_d = (x[:, -1] - x[:, 0]) 
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # [N, 2]
        z_vals = torch.linspace(0.0, 2.0, T, device=device).unsqueeze(0) #[1, T]
        z_vals = z_vals.expand((B*N, T)) # [N, T]
        
        sample_dist = 1. / T

        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(0, 1) # avoid out of bounds xyzs.

        xys = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) #  [N, T, 2]
        aabb = torch.tensor([-1, -1, 1, 1], dtype=torch.float32).to(device)
        xys = torch.min(torch.max(xys, aabb[:2]), aabb[2:]) # a manual clip.
        sigma_outputs = self.sigma(xys.reshape(-1, 2))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
        deltas = torch.cat([sample_dist * torch.ones_like(deltas[..., :1]), deltas], dim=-1) # [N, T]

        sigma = sigma_outputs['sigma'].view(-1, T)
        density = deltas * sigma
        density = density.view(B, N, T)

        return density # [B, N, T]
    
    def sigma(self, x):
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = self.act(h)
        
        # sigma = torch.exp(h)
        sigma = self.act(h)

        return {
            'sigma': sigma
        }