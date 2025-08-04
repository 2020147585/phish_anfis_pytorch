#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from anfis import AnfisNet
from sklearn.cluster import KMeans
import numpy as np

def _mk_param(val):
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMembFunc(torch.nn.Module):
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))
    def forward(self, x):
        return torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
    def pretty(self):
        return f'GaussMembFunc mu={self.mu.item()} sigma={self.sigma.item()}'

class BellMembFunc(torch.nn.Module):
    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))
    def pretty(self):
        return f'BellMembFunc a={self.a.item()} b={self.b.item()} c={self.c.item()}'

def make_gauss_mfs(sigma, mu_list):
    return [GaussMembFunc(mu, sigma) for mu in mu_list]

def make_bell_mfs(a, b, clist):
    return [BellMembFunc(a, b, c) for c in clist]

def get_kmeans_centers(feature_values, num_mfs):
    unique_values = np.unique(feature_values)
    clusters = min(num_mfs, len(unique_values))
    km = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    km.fit(feature_values.reshape(-1, 1))
    return sorted(km.cluster_centers_.flatten())

def make_anfis(x, num_mfs=3, hybrid=True, mf_type='gauss', use_kmeans=True):
    num_invars = x.shape[1]
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    ranges = maxvals - minvals

    invars = []
    for i in range(num_invars):
        feature_vals = x[:, i].numpy()
        if use_kmeans:
            centers = get_kmeans_centers(feature_vals, num_mfs)
        else:
            centers = torch.linspace(minvals[i], maxvals[i], num_mfs).tolist()

        if mf_type == 'gauss':
            sigma = ranges[i] / (num_mfs if num_mfs > 0 else 1)
            mfs = make_gauss_mfs(sigma, centers)
        elif mf_type == 'bell':
            a = ranges[i] / (2 * num_mfs)
            b = 2.0
            mfs = make_bell_mfs(a, b, centers)
        else:
            raise ValueError("Unsupported MF type. Use 'gauss' or 'bell'.")

        invars.append((f'x{i}', mfs))

    outvars = ['y']
    model = AnfisNet('Phishing classifier', invars, outvars, hybrid=hybrid)
    return model
