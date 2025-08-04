import pandas as pd
import torch
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from membership import make_anfis
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
import wandb


parser = argparse.ArgumentParser(description="Train ANFIS model with wandb and argparse")
parser.add_argument('--dataset', type=str, default='urlset', choices=['phishStorm', 'ISCX-URL-2016', 'DEPHIDES'], help='please choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--epochs', type=int, default=50, help='epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
parser.add_argument('--num_mfs', type=int, default=3, help='mfs_number')
parser.add_argument('--mf_type', type=str, default='bell', choices=['bell', 'gauss'], help='gauss or bell')
parser.add_argument('--scheduler_step', type=int, default=20, help='StepLR')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Learning rate attenuation coefficient')
args = parser.parse_args()


wandb.init(project="anfis-phishing", name="anfis_run_argparse")
wandb.config.update(vars(args))

if args.dataset == 'phishStorm':
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
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

#Model, loss function, optimizer
model = make_anfis(X_train, num_mfs=args.num_mfs, hybrid=False, mf_type=args.mf_type, use_kmeans=True)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

metrics = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}

# LOOP
for epoch in range(args.epochs):
    model.train()
    for xb, yb in train_loader:
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    with torch.no_grad():
        model.eval()
        y_pred_test = model(X_test)
        preds = (y_pred_test > 0.5).float()
        acc = (preds == y_test).float().mean().item()
        precision = precision_score(y_test.numpy(), preds.numpy())
        recall = recall_score(y_test.numpy(), preds.numpy())
        f1 = f1_score(y_test.numpy(), preds.numpy())
        auc = roc_auc_score(y_test.numpy(), y_pred_test.numpy())

        metrics["Loss"].append(loss.item())
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["F1"].append(f1)
        metrics["AUC"].append(auc)

        wandb.log({
            "epoch": epoch + 1,
            "loss": loss.item(),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "learning_rate": current_lr
        })

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}, "
              f"Acc: {acc:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, "
              f"LR: {current_lr:.6f}")

#LSE adjustment
with torch.no_grad():
    model.fit_coeff(X_train, y_train)

# Draw the index curve
epochs_range = range(1, args.epochs + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, metrics["Accuracy"], label="Accuracy")
plt.plot(epochs_range, metrics["Precision"], label="Precision")
plt.plot(epochs_range, metrics["Recall"], label="Recall")
plt.plot(epochs_range, metrics["F1"], label="F1-score")
plt.plot(epochs_range, metrics["AUC"], label="AUC")
plt.title("Model Metrics Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.savefig("anfis_metrics_curve.png")
plt.close()
wandb.save("anfis_metrics_curve.png")

#prediction vs. true distribution map
true_counts = np.bincount(y_test.numpy().flatten().astype(int))
pred_counts = np.bincount(preds.numpy().flatten().astype(int))
labels = [0, 1]
x = np.arange(len(labels))

plt.bar(x - 0.2, true_counts, width=0.4, label='True')
plt.bar(x + 0.2, pred_counts, width=0.4, label='Predicted')
for i, v in enumerate(true_counts):
    plt.text(i - 0.2, v + 50, str(v), ha='center', fontsize=10)
for i, v in enumerate(pred_counts):
    plt.text(i + 0.2, v + 50, str(v), ha='center', fontsize=10)

plt.xticks(x, labels)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("True vs Predicted Label Distribution")
plt.legend()
plt.savefig("anfis_predictions_vs_true_bar.png")
plt.close()
wandb.save("anfis_predictions_vs_true_bar.png")

# final assenment
with torch.no_grad():
    model.eval()
    y_pred_test = model(X_test)

    fpr_curve, tpr_curve, thresholds = roc_curve(y_test.numpy(), y_pred_test.numpy())
    optimal_idx = np.argmax(tpr_curve - fpr_curve)
    optimal_threshold = thresholds[optimal_idx]
    preds = (y_pred_test > optimal_threshold).float()
    acc = (preds == y_test).float().mean().item()

    cm = confusion_matrix(y_test.numpy(), preds.numpy())
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    with open("anfis_final_report.txt", "w") as f:
        f.write(f"Final Accuracy: {acc:.4f}\n")
        f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
        f.write(f"False Negative Rate (FNR): {fnr:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
    wandb.save("anfis_final_report.txt")

# save False positive rate and false negative rate and threshold
fnr_curve = 1 - tpr_curve
plt.figure(figsize=(8, 5))
plt.plot(thresholds, fpr_curve, label="False Positive Rate")
plt.plot(thresholds, fnr_curve, label="False Negative Rate")
plt.axvline(optimal_threshold, color='red', linestyle='--', label='Optimal Threshold')
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("FPR and FNR vs Threshold")
plt.legend()
plt.savefig("fpr_fnr_vs_threshold.png")
plt.close()
wandb.save("fpr_fnr_vs_threshold.png")

#save consusion matrix
labels = np.array([
    [f"TN\n{tn}", f"FP\n{fp}"],
    [f"FN\n{fn}", f"TP\n{tp}"]
])
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
plt.colorbar(im)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, labels[i, j], ha='center', va='center', color='red', fontsize=12)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Negative (0)", "Positive (1)"])
ax.set_yticklabels(["Negative (0)", "Positive (1)"])
plt.savefig("anfis_confusion_matrix_labeled.png")
plt.close()
wandb.save("anfis_confusion_matrix_labeled.png")

#save modle
torch.save(model.state_dict(), "anfis_model.pth")
wandb.save("anfis_model.pth")
