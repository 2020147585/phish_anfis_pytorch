# ANFIS-Based Phishing URL Detection (PyTorch)

This repository implements a **PyTorch-based Adaptive Neuro-Fuzzy Inference System (ANFIS)** for phishing URL detection.  
It supports **Weights & Biases (W&B)** visualization, flexible **command-line arguments**, and outputs comprehensive performance metrics.

---

## New Features (compared to previous version)
- **Threshold Optimization**: automatically selects the best `best_threshold`, configurable via `--optimize_for f1/accuracy/precision/recall`.
- **Extended Final Metrics**:
  - Accuracy, Precision, Recall, F1, False Positive Rate
  - `inference_time_ms_per_instance` (per-URL inference time)
  - `avg_mem_mb_single_url` / `max_mem_mb_single_url` (memory usage fluctuation in MB)
  - `python_peak_alloc_kb` (Python-level peak allocation in KB)
  - `model_size_mb` (model file size)
  - `strong_conf_*` metrics (coverage and accuracy of high-confidence predictions)
- **W&B Line Logging**: Accuracy, Precision, Recall, F1, and FPR tracked per epoch.
- **Result folder (`result/`)** with JSON metrics and final reports:
  - `anfis_metrics_<dataset>.json` (final metrics)
  - `anfis_final_report.txt` (detailed confusion matrix & error rates)

---

## Features
- ANFIS implementation in PyTorch
- Supports **bell-shaped** and **Gaussian** membership functions
- K-Means initialization
- **Hybrid Learning** (Backpropagation + Least Squares Estimation)
- StepLR learning rate scheduler
- Dataset support: ISCX-URL-2016, PhishStorm, DEPHIDES
- Real-time W&B monitoring (epoch curves + final metrics)

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

---

##  How to Run

```bash
python phishing_example.py     --dataset ISCX-URL-2016     --batch_size 64     --epochs 50     --lr 0.001     --num_mfs 3     --mf_type bell     --scheduler_step 20     --scheduler_gamma 0.5     --optimize_for f1     --strong_pos_threshold 0.9     --strong_neg_threshold 0.1
```

### Command-line Arguments
| Argument                  | Default   | Description |
|---------------------------|-----------|-------------|
| `--dataset`              | urlset    | Dataset: 'phishStorm', 'ISCX-URL-2016', 'DEPHIDES', 'urlset' |
| `--batch_size`           | 64        | Training batch size |
| `--epochs`               | 50        | Number of epochs |
| `--lr`                   | 0.001     | Initial learning rate |
| `--num_mfs`              | 3         | Membership functions per feature |
| `--mf_type`              | bell      | Membership type: `bell`, `gauss` |
| `--scheduler_step`       | 20        | StepLR step size |
| `--scheduler_gamma`      | 0.5       | StepLR decay factor |
| `--optimize_for`         | f1        | Metric to optimize threshold: f1 / accuracy / precision / recall |
| `--strong_pos_threshold` | 0.9       | Strong positive confidence threshold |
| `--strong_neg_threshold` | 0.1       | Strong negative confidence threshold |

---

##  Training Results

### Per-Epoch Metrics (W&B Line Charts)
- Accuracy, Precision, Recall, F1, False Positive Rate  

### Final Output Metrics (JSON & W&B summary)
```json
{
  "accuracy": 0.8884189980481457,
  "precision": 0.8939597315436242,
  "recall": 0.8780487804878049,
  "f1": 0.8859328234120386,
  "false_positive_rate": 0.1014771997430957,
  "inference_time_ms_per_instance": 1.527568832042192,
  "avg_mem_mb_single_url": 0.0042266845703125,
  "max_mem_mb_single_url": 0.0390625,
  "python_peak_alloc_kb": 4.3115234375,
  "model_size_mb": 0.013763427734375,
  "best_threshold": 0.45,
  "optimize_for": "f1",
  "strong_pos_threshold": 0.9,
  "strong_neg_threshold": 0.1,
  "strong_pos_count": 981,
  "strong_neg_count": 821,
  "strong_conf_total": 1802,
  "strong_conf_coverage": 0.5862068965517241,
  "strong_pos_accuracy": 0.9857288481141692,
  "strong_neg_accuracy": 0.9646772228989038
}
```

### Visualizations
- `anfis_metrics_curve.png` → Metric curves  
- `anfis_predictions_vs_true_bar.png` → Label distribution comparison  
- `anfis_confusion_matrix_labeled.png` → Confusion matrix  
- `fpr_fnr_vs_threshold.png` → FPR/FNR vs threshold curve  

---

##  Output Files
- `result/anfis_metrics_<dataset>.json` → Final metrics  
- `anfis_final_report.txt` → Accuracy / FPR / FNR / Confusion Matrix   
- Visualizations (curves, bars, confusion matrix, etc.)  

---

##  W&B Dashboard
Training and results can be viewed in W&B:  
[View Project](https://wandb.ai/YOUR_USERNAME/anfis-phishing)

---

##  Known Limitations / Future Work
- Weaker performance on PhishStorm / DEPHIDES datasets → requires better feature engineering  
- Limited high-confidence coverage → needs more membership functions and structural tuning  
- Currently only supports bell / gauss membership functions → future: triangular / custom MFs  
- Hyperparameters are manually tuned → plan: Optuna / AutoML for optimization  

