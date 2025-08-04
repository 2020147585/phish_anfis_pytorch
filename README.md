# phish_anfis_pytorch
# ANFIS-Based Phishing URL Detection

This project implements an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** model for phishing URL detection using **PyTorch**.  
It supports **Weights & Biases (W&B)** for training visualization and **command-line arguments (argparse)** for flexible configuration.

---

##  Features
-  PyTorch implementation of ANFIS  
-  Supports both **bell-shaped** and **Gaussian** membership functions  
-  Automatic initialization with **K-Means clustering**  
-  **Hybrid Learning**: Backpropagation + Least Squares Estimation (LSE)  
-  **Dynamic learning rate** scheduling via `StepLR`  
-  Full **training visualization** with W&B  
-  Saves training metrics, confusion matrix, final model weights，etc..  
-  Multiple dataset support (ISCX-URL-2016, PhishStorm, DEPHIDE)

---

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/2020147585/phish_anfis_pytorch.git
cd phish_anfis_pytorch
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Login to W&B (Optional)
```bash
wandb login
```
> If you don't have an account, choose "Create a W&B account" when prompted.

---

## How to Run

```bash
python phishing_example.py \
    --dataset ISCX-URL-2016 \
    --batch_size 64 \
    --epochs 50 \
    --lr 0.001 \
    --num_mfs 3 \
    --mf_type bell \
    --scheduler_step 20 \
    --scheduler_gamma 0.5
```
(there has a ouput that can be login wandb if you want)
### Command-line Arguments
| Parameter         | Default | Description |
|-------------------|---------|-------------|
| `--dataset`        | urlset  | Dataset type: 'phishStorm', 'ISCX-URL-2016', 'DEPHIDES' |
| `--batch_size`     | 64      | Training batch size |
| `--epochs`         | 50      | Number of training epochs |
| `--lr`             | 0.001   | Initial learning rate |
| `--num_mfs`        | 3       | Number of membership functions per feature |
| `--mf_type`        | bell    | Membership function type: `bell`, `gauss` |
| `--scheduler_step` | 20      | StepLR: Reduce LR every N epochs |
| `--scheduler_gamma`| 0.5     | StepLR: Multiplicative factor for LR |

---

##  Training Results

###  Metrics (Accuracy, Precision, Recall, F1, AUC)

###  Prediction vs True Label Distribution


###  False Positive/Negative Rate vs Threshold

###  Confusion Matrix

###  Final Report.txt

```
Final Accuracy
False Positive Rate (FPR)
False Negative Rate (FNR)
Confusion Matrix:
[[TN FP]
 [FN TP]]
```

---

## Output Files
- `anfis_metrics_curve.png` → Training metrics plot  
- `anfis_predictions_vs_true_bar.png` → Prediction distribution  
- `fpr_fnr_vs_threshold.png` → Threshold optimization curve  
- `anfis_confusion_matrix_labeled.png` → Confusion matrix visualization  
- `anfis_final_report.txt` → Final report with Accuracy, FPR, FNR  
- `anfis_model.pth` → Trained model weights  

---

##  W&B Dashboard
You can view detailed logs and training visualizations on W&B if you login before training:  
[View Project](https://wandb.ai/YOUR_USERNAME/anfis-phishing)

---

 

---

##  Known Limitations / Future Improvements
-  Accuracy can decrease when adding more features → Need **feature selection** or **regularization**  
-  Hybrid mode may freeze training → Requires improved LSE optimization  
-  Membership functions are fixed (Bell/Gauss only) → Future: add **triangular** or **custom MFs**  
-  Hyperparameters tuned manually → Future: integrate **AutoML or Optuna**  
-  Model currently tested on limited datasets. Moreover, for the PhishStorm and DEPHIDES datasets, although the data volume is large, the results are not good enough. The reason might lie in some errors made during data cleaning → Need more phishing datasets for robustness  

---

