#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

dtype = torch.float

def plotErrors(errors):
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('Percentage error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plotResults(y_actual, y_predicted):
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()

def _plot_mfs(var_name, fv, x):
    xsort, _ = x.sort()
    for mfname, yvals in fv.fuzzify(xsort):
        plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
    plt.xlabel(f'Values for variable {var_name} ({fv.num_mfs} MFs)')
    plt.ylabel('Membership')
    plt.legend(bbox_to_anchor=(1., 0.95))
    plt.show()

def plot_all_mfs(model, x):
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        _plot_mfs(var_name, fv, x[:, i])

def calc_error(y_pred, y_actual):
    with torch.no_grad():
        loss = F.binary_cross_entropy(y_pred, y_actual)
        preds = (y_pred > 0.5).float()
        acc = (preds == y_actual).float().mean().item()
    return loss.item(), acc

def test_anfis(model, data, show_plots=False):
    x, y_actual = data.dataset.tensors
    if show_plots:
        plot_all_mfs(model, x)
    print(f'### Testing for {x.shape[0]} cases')
    y_pred = model(x)
    loss, acc = calc_error(y_pred, y_actual)
    print(f'BCELoss={loss:.5f}, Accuracy={acc:.4f}')
    if show_plots:
        plotResults(y_actual, y_pred)

def train_anfis_with(model, data, optimizer, criterion, epochs=500, show_plots=False):
    errors = []
    print(f'### Training for {epochs} epochs, training size = {data.dataset.tensors[0].shape[0]} cases')
    for t in range(epochs):
        for x, y_actual in data:
            y_pred = model(x)
            loss = criterion(y_pred, y_actual)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        x_all, y_all = data.dataset.tensors
        with torch.no_grad():
            model.fit_coeff(x_all, y_all)
            y_pred_all = model(x_all)
            loss_val, acc = calc_error(y_pred_all, y_all)
            errors.append(loss_val)
        if epochs < 30 or t % 10 == 0:
            print(f'Epoch {t:4d}: Loss={loss_val:.5f}, Accuracy={acc:.4f}')
    if show_plots:
        plotErrors(errors)
        y_pred_all = model(data.dataset.tensors[0])
        plotResults(data.dataset.tensors[1], y_pred_all)

def train_anfis(model, data, epochs=500, show_plots=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    train_anfis_with(model, data, optimizer, criterion, epochs, show_plots)
