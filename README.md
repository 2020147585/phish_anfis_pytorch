# phish_anfis_pytorch
# ANFIS-Based Phishing URL Detection

This project implements an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** model for phishing URL detection using **PyTorch**.  
It supports **Weights & Biases (W&B)** for training visualization and **command-line arguments (argparse)** for flexible configuration.

---

## Features
-  PyTorch implementation of ANFIS  
-  Supports both **bell-shaped** and **Gaussian** membership functions  
-  Automatic initialization with **K-Means clustering**  
-  **Hybrid Learning**: Backpropagation + Least Squares Estimation (LSE)  
-  **Dynamic learning rate** scheduling via `StepLR`  
-  Full **training visualization** with W&B  
-  Saves training metrics, confusion matrix, final model weights,etc..  
-  Multiple dataset support (ISCX-URL-2016, PhishStorm, DEPHIDES)

---

## 🛠️ Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/2020147585/phish_anfis_pytorch.git
cd anfis-phishing
