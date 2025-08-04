#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

dtype = torch.float


class FuzzifyVariable(torch.nn.Module):
    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):
            mfnames = [f'mf{i}' for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        return len(self.mfdefs)

    def members(self):
        return self.mfdefs.items()

    def pad_to(self, new_size):
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield(mfname, yvals)

    def forward(self, x):
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred

class FuzzifyLayer(torch.nn.Module):
    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = [f'x{i}' for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        return len(self.varmfs)

    @property
    def max_mfs(self):
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append(f'Variable {varname}')
            for mfname, mfdef in members.mfdefs.items():
                params = ', '.join([f'{n}={p.item()}' for n, p in mfdef.named_parameters()])
                r.append(f'- {mfname}: {mfdef.__class__.__name__}({params})')
        return '\n'.join(r)

    def forward(self, x):
        assert x.shape[1] == self.num_in
        y_pred = torch.stack([var(x[:, i:i+1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred

class AntecedentLayer(torch.nn.Module):
    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        mf_count = [var.num_mfs for var in varlist]
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))

    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append(f'{varname} is {list(fv.mfdefs.keys())[i]}')
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        rules = torch.prod(ants, dim=2)
        return rules

class ConsequentLayer(torch.nn.Module):
    def __init__(self, d_in, d_rule, d_out):
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in+1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        assert new_coeff.shape == self.coeff.shape
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        weighted_x[weighted_x == 0] = 1e-12
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        coeff_2d, _ = torch.gels(y_actual_2d, weighted_x_2d)
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1]+1, -1).transpose(1, 2)

    def forward(self, x):
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)

class PlainConsequentLayer(ConsequentLayer):
    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients', torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        raise AssertionError('Not hybrid learning: using BP to learn coefficients')

class WeightedSumLayer(torch.nn.Module):
    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)


class AnfisNet(torch.nn.Module):
    def __init__(self, description, invardefs, outvarnames, hybrid=True):
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.hybrid = hybrid
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])

        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)

        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs)),
            ('consequent', cl),
        ]))

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        return self.outvarnames

    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append(f'Rule {i:2d}: IF {rule_ants[i]}')
            rstr.append(' ' * 9 + f'THEN {crow.tolist()}')
        return '\n'.join(rstr)

    def forward(self, x):
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2)).squeeze(2)
        y_pred = torch.sigmoid(y_pred)
        self.y_pred = y_pred
        return self.y_pred
