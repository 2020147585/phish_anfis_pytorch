
import os, time, json, gc, argparse
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

import wandb
from membership import make_anfis

parser = argparse.ArgumentParser(description="Train ANFIS and log VAE-DNN-style metrics")
parser.add_argument('--dataset', type=str, default='urlset',
                    choices=['phishStorm', 'ISCX-URL-2016', 'DEPHIDES', 'urlset'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_mfs', type=int, default=3)
parser.add_argument('--mf_type', type=str, default='bell', choices=['bell', 'gauss'])
parser.add_argument('--scheduler_step', type=int, default=20)
parser.add_argument('--scheduler_gamma', type=float, default=0.5)

parser.add_argument('--optimize_for', type=str, default='f1',
                    choices=['f1', 'accuracy', 'precision', 'recall'])
parser.add_argument('--strong_pos_threshold', type=float, default=0.9)
parser.add_argument('--strong_neg_threshold', type=float, default=0.1)


args = parser.parse_args()


wandb.init(project="anfis-phishing", name=f"anfis_lines_{args.dataset}")
wandb.config.update(vars(args))


if args.dataset in ('phishStorm', 'urlset'):
    df = pd.read_csv("urlset_cleaned.csv")
    selected_cols = ['card_rem', 'mld_res', 'ranking', 'ratio_Arem']
    X = df[selected_cols].values
    y = df['label'].values.reshape(-1, 1)

elif args.dataset == 'ISCX-URL-2016':
    df = pd.read_csv("Phishing_Infogain.csv")
    df['class'] = LabelEncoder().fit_transform(df['class'])
    selected_cols = ['domain_token_count', 'domainUrlRatio', 'NumberofDotsinURL', 'domainlength']
    X = df[selected_cols].values
    y = df['class'].values.reshape(-1, 1)

elif args.dataset == 'DEPHIDES':
    df = pd.read_csv("val_features.csv")
    selected_cols = ['NumberofDotsinURL', 'LongestPathTokenLength', 'domain_token_count', 'avgdomaintokenlen']
    X = df[selected_cols].values
    y = df['class'].values.reshape(-1, 1)


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor.numpy().ravel()
)


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train.numpy().ravel()
)


model = make_anfis(X_train, num_mfs=args.num_mfs, hybrid=False, mf_type=args.mf_type, use_kmeans=True)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)


def _ensure_bin(y):
    return np.asarray(y).reshape(-1).astype(int)

def metrics_at_th(y_true, y_prob, th):
    y_true = _ensure_bin(y_true)
    y_pred = (np.asarray(y_prob).reshape(-1) >= th).astype(int)
    acc  = float((y_pred == y_true).mean())
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1v  = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = float(fp) / float(max(1, (fp + tn)))
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1v, 'false_positive_rate': fpr}

def scan_best_threshold(model, Xv, yv, optimize_for='f1', num_points=91):
    with torch.no_grad():
        pv = model(Xv).numpy().reshape(-1)
    ths = np.linspace(0.05, 0.95, num_points)
    best_th, best_val, best_m = 0.5, -1.0, None
    for th in ths:
        m = metrics_at_th(yv.numpy(), pv, th)
        score = m[optimize_for]
        if score > best_val:
            best_val, best_th, best_m = score, float(th), m
    return best_th, best_m


for epoch in range(1, args.epochs + 1):
    model.train()
    for xb, yb in train_loader:
        yhat = model(xb)
        loss = criterion(yhat, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    
    best_th_e, _ = scan_best_threshold(model, X_val, y_val, optimize_for=args.optimize_for)
    with torch.no_grad():
        pt = model(X_test).numpy().reshape(-1)
    m = metrics_at_th(y_test.numpy(), pt, best_th_e)

    

    wandb.log({
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "false_positive_rate": m["false_positive_rate"],
        "epoch": epoch
    }, step=epoch)

    print(f"[Epoch {epoch:3d}] acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
          f"rec={m['recall']:.4f} f1={m['f1']:.4f} fpr={m['false_positive_rate']:.4f} "
          f"(best_th={best_th_e:.2f})")



with torch.no_grad():
    model.fit_coeff(X_train, y_train)




best_threshold, _ = scan_best_threshold(model, X_val, y_val, optimize_for=args.optimize_for)
with torch.no_grad():
    yprob_test = model(X_test).numpy().reshape(-1)
final_metrics = metrics_at_th(y_test.numpy(), yprob_test, best_threshold)




import threading, psutil, time, numpy as np, os, gc, torch

def _sample_uss_loop(proc, stop_event, interval_sec, holder_dict):
    max_uss = holder_dict.get("max_uss", 0)
    while not stop_event.is_set():
        try:
            uss = proc.memory_full_info().uss
            if uss > max_uss:
                max_uss = uss
        except Exception:
            pass
        time.sleep(interval_sec)
    holder_dict["max_uss"] = max_uss

def measure_per_url_time_and_mem(model, X_tensor, n_urls=256, sample_every_ms=2.0, warmup=8):

    proc = psutil.Process(os.getpid())
    n = min(len(X_tensor), n_urls)


    for i in range(min(warmup, len(X_tensor))):
        with torch.no_grad():
            _ = model(X_tensor[i].unsqueeze(0))

    gc.collect()

    deltas_mb = []
    per_url_ms = []

    for i in range(n):
        x = X_tensor[i].unsqueeze(0)


        try:
            uss_before = proc.memory_full_info().uss
        except Exception:
            uss_before = proc.memory_info().rss

        holder = {"max_uss": uss_before}
        stop_event = threading.Event()
        t = threading.Thread(
            target=_sample_uss_loop,
            args=(proc, stop_event, max(0.001, sample_every_ms/1000.0), holder),
            daemon=True
        )
        t.start()


        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        t1 = time.perf_counter()


        stop_event.set()
        t.join()


        uss_peak = holder.get("max_uss", uss_before)
        delta_mb = max(0.0, (uss_peak - uss_before) / (1024.0 ** 2))
        deltas_mb.append(float(delta_mb))

        per_url_ms.append((t1 - t0) * 1000.0)

    infer_time_ms_per_instance = float(np.mean(per_url_ms)) if per_url_ms else 0.0
    avg_mb = float(np.mean(deltas_mb)) if deltas_mb else 0.0
    max_mb = float(np.max(deltas_mb)) if deltas_mb else 0.0
    return infer_time_ms_per_instance, avg_mb, max_mb


infer_ms, avg_mem_mb_single_url, max_mem_mb_single_url = measure_per_url_time_and_mem(
    model, X_test, n_urls=256, sample_every_ms=2.0, warmup=8
)



def predict_one(x_row_tensor):
    with torch.no_grad():
        return float(model(x_row_tensor.unsqueeze(0)).item())


import tracemalloc

def measure_tracemalloc_peak_kb(predict_one_fn, X_tensor, n_samples=128, warmup=10):
    for i in range(min(warmup, len(X_tensor))):
        _ = predict_one_fn(X_tensor[i])

    tracemalloc.start()
    for i in range(min(n_samples, len(X_tensor))):
        _ = predict_one_fn(X_tensor[i])
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(peak / 1024.0)  # KB
peak_kb = measure_tracemalloc_peak_kb(predict_one, X_test, n_samples=256)



model_path = "anfis_model.pth"
torch.save(model.state_dict(), model_path)
model_size_mb = os.path.getsize(model_path) / (1024 ** 2)


sp_th = float(args.strong_pos_threshold)
sn_th = float(args.strong_neg_threshold)
y_true_np = _ensure_bin(y_test.numpy())
y_pred_lock = (yprob_test >= best_threshold).astype(int)

mask_sp = yprob_test >= sp_th
mask_sn = yprob_test <= sn_th

acc_sp = float((y_true_np[mask_sp] == y_pred_lock[mask_sp]).mean()) if mask_sp.any() else None
acc_sn = float((y_true_np[mask_sn] == y_pred_lock[mask_sn]).mean()) if mask_sn.any() else None

sp_count = int(mask_sp.sum())
sn_count = int(mask_sn.sum())
strong_total = sp_count + sn_count
strong_cov = strong_total / len(y_true_np)



full_metrics = {
    "accuracy": final_metrics["accuracy"],
    "precision": final_metrics["precision"],
    "recall": final_metrics["recall"],
    "f1": final_metrics["f1"],
    "false_positive_rate": final_metrics["false_positive_rate"],
    "inference_time_ms_per_instance": infer_ms,
    "avg_mem_mb_single_url": avg_mem_mb_single_url,     
    "max_mem_mb_single_url": max_mem_mb_single_url,
    "python_peak_alloc_kb": peak_kb,
    "model_size_mb": model_size_mb,
    "best_threshold": float(round(best_threshold, 2)),
    "optimize_for": args.optimize_for,
    "strong_pos_threshold": sp_th,
    "strong_neg_threshold": sn_th,
    "strong_pos_count": sp_count,
    "strong_neg_count": sn_count,
    "strong_conf_total": strong_total,
    "strong_conf_coverage": float(strong_cov),
    "strong_pos_accuracy": float(acc_sp) if acc_sp is not None else None,
    "strong_neg_accuracy": float(acc_sn) if acc_sn is not None else None
}


os.makedirs("result", exist_ok=True)
json_path = f"result/anfis_metrics_{args.dataset}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(full_metrics, f, indent=2, ensure_ascii=False)
print(f"[Saved] {json_path}")


log_metrics = {
    "accuracy": full_metrics["accuracy"],
    "precision": full_metrics["precision"],
    "recall": full_metrics["recall"],
    "f1": full_metrics["f1"],
    "false_positive_rate": full_metrics["false_positive_rate"],
    "inference_time_ms_per_instance": infer_ms,
    "avg_mem_mb_single_url": avg_mem_mb_single_url,     
    "max_mem_mb_single_url": max_mem_mb_single_url,
    "model_size_mb": model_size_mb,
}


wandb.run.summary.update(log_metrics)
wandb.log(log_metrics)
wandb.finish()
