# Study of Adaptive Gradient Methods in NLP
### Project 16 — Optimization Algorithms Module | Master DSBD & AI | 2025-2026

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

---

## 📌 Overview

This repository contains the full implementation, experimental results, and academic report for **Project 16** of the Optimization Algorithms module (Master DSBD & AI, 2025-2026).

The project presents a rigorous empirical comparison of **five gradient descent optimization algorithms** applied to fine-tuning a **DistilBERT** model on the **IMDB binary sentiment classification** task. The study analyzes convergence speed, training stability, learning rate sensitivity, batch size robustness, GPU memory efficiency, and layer-wise gradient norm dynamics.

---

## 🎯 Research Question

> *Which optimization algorithm offers the best trade-off between classification performance, training stability, and computational efficiency when fine-tuning a Transformer model on an NLP task?*

---

## 🔬 Optimizers Compared

| Optimizer | Type | Key Feature |
|---|---|---|
| **SGD + Momentum** | Non-adaptive | Uniform learning rate baseline (γ=0.9) |
| **AdamW** | Adaptive | Decoupled weight decay regularization |
| **AdaFactor** | Adaptive | Sublinear memory via matrix factorization |
| **LAMB** | Adaptive | Layer-wise trust ratio for large batches |
| **Lion** | Adaptive | Sign-based momentum (EvoLved optimizer) |

---

## 📊 Key Results

### Main Benchmark (Test Set — 3 Epochs)

| Optimizer | Accuracy | F1-Score | GPU Memory | Training Time | Rank |
|---|---|---|---|---|---|
| **AdaFactor** | **89.65%** | **89.72%** | **1204 MB** | 263.5s | 🥇 1 |
| AdamW | 89.50% | 89.53% | 1724 MB | 250.6s | 🥈 2 |
| Lion | 85.65% | 85.99% | 1468 MB | 243.8s | 🥉 3 |
| LAMB | 81.70% | 81.52% | 1719 MB | 273.3s | 4 |
| SGD | 53.35% | 53.79% | 1458 MB | 235.6s | 5 |

> ✅ **AdaFactor achieves the best overall trade-off: highest F1-score (89.72%) with 30% less GPU memory than AdamW.**

### Learning Rate Sensitivity (1 Epoch, 4 LR values)

| Optimizer | Best LR | Best F1 | Stability |
|---|---|---|---|
| AdamW | 5e-5 | 0.8980 | Excellent |
| AdaFactor | 5e-5 | 0.8971 | Very Good |
| Lion | 2e-5 | 0.9052 | Medium |
| LAMB | 1e-4 | 0.8687 | Good |
| SGD | 1e-4 | 0.5306 | Unstable |

### Batch Size Robustness (1 Epoch, 3 Batch Sizes)

| Optimizer | BS=8 F1 | BS=16 F1 | BS=32 F1 | Memory Range |
|---|---|---|---|---|
| AdaFactor | 0.8977 | 0.8946 | 0.8899 | 793–2021 MB |
| AdamW | 0.8958 | 0.8941 | 0.8927 | 1333–2551 MB |
| Lion | 0.8899 | 0.9052 | 0.8958 | 1081–2288 MB |
| LAMB | 0.6779 | 0.6252 | 0.5429 | 1336–2551 MB |
| SGD | 0.3315 | 0.4176 | 0.4335 | 1076–2287 MB |

---

## 🗂️ Repository Structure

```
projet16-adaptive-gradient-methods-nlp/
│
├── 📓 FINALE_Study_of_Adaptive_Gradient_Methods_in_NLP.ipynb
│       └── Complete reproducible benchmark notebook (Google Colab / Jupyter)
│
├── 📄 Projet16_Adaptive_Gradient_Methods_Final_Report.docx
│       └── Full academic report (~14 pages, English)
│
├── 📄 Projet16_Adaptive_Gradient_Methods_Final_Report.pdf
│       └── PDF version of the academic report
│
├── results/
│   ├── main_benchmark_results.csv       ← 5 optimizers × 9 metrics
│   ├── lr_sensitivity_results.csv       ← 5 optimizers × 4 learning rates
│   ├── batch_size_results.csv           ← 5 optimizers × 3 batch sizes
│   ├── final_summary.xlsx               ← Excel workbook (3 sheets)
│   ├── README.md                        ← Results documentation
│   └── figures/
│       ├── loss_curves.png              ← Train & validation loss per epoch
│       ├── metric_curves.png            ← Accuracy & F1-score per epoch
│       ├── gradient_norms_trajectory.png ← Global gradient norm evolution
│       ├── layerwise_gradient_norms.png ← Layer-wise gradient norms (Epoch 3)
│       ├── final_metrics_comparison.png ← Bar charts: accuracy, F1, time, memory
│       ├── confusion_matrix.png         ← AdaFactor confusion matrix (test set)
│       ├── lr_sensitivity.png           ← LR sensitivity curves
│       ├── batch_size_stability.png     ← Batch size robustness curves
│       ├── dataset_exploration.png      ← Label distribution & token lengths
│       ├── schema_global_arch.png       ← Global NLP architecture schema
│       ├── schema_exp_pipeline.png      ← Experimental pipeline schema
│       └── schema_grad_layers.png       ← Layer-wise gradient extraction schema
│
└── README.md
```

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Upload `FINALE_Study_of_Adaptive_Gradient_Methods_in_NLP.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially (`Runtime → Run all`)
4. All results are automatically saved to `results/`

### Option 2 — Local Jupyter

```bash
# Clone the repository
git clone https://github.com/BOULATARM/projet16-adaptive-gradient-methods-nlp.git
cd projet16-adaptive-gradient-methods-nlp

# Install dependencies
pip install transformers==4.52.4 datasets==3.6.0 torch torchvision
pip install scikit-learn pandas matplotlib openpyxl pytorch-optimizer

# Launch notebook
jupyter notebook FINALE_Study_of_Adaptive_Gradient_Methods_in_NLP.ipynb
```

---

## ⚙️ Experimental Protocol

| Hyperparameter | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Task | Binary Sentiment Classification |
| Dataset | IMDB (8k train / 2k val / 2k test) |
| Random Seed | 42 |
| Number of Epochs | 3 |
| Batch Size | 16 |
| Initial Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Max Sequence Length | 256 tokens |
| Warmup Ratio | 10% |
| Mixed Precision | AMP (FP16) |
| Gradient Clipping | max_norm=1.0 |

---

## 🏗️ Technical Architecture

- **Model** : DistilBERT (`distilbert-base-uncased`) — 66M parameters, 6 Transformer encoder layers
- **Tokenizer** : WordPiece subword tokenization, max_length=256
- **Training** : Custom PyTorch loop with AMP (`torch.amp.autocast('cuda')`)
- **Fallbacks** : Native PyTorch implementations of Lion and LAMB
- **Tracking** : Layer-wise gradient norms (Embeddings, Layer 0, Layer 2, Layer 5, Classifier)
- **Evaluation** : Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## 📈 Main Findings

1. **AdaFactor** achieves the best overall trade-off — top F1 (89.72%) with 30% less GPU memory than AdamW, due to its factored second-moment tracking.

2. **AdamW** remains a strong and stable baseline — excellent LR robustness, fast convergence, standard choice for Transformer fine-tuning.

3. **Lion** is competitive but sensitive to learning rate — best F1 at LR=2e-5 (90.52% in LR study), but exhibits overfitting at epoch 3.

4. **LAMB** underperforms at small batch sizes — its conservative trust ratio slows convergence within a 3-epoch budget.

5. **SGD** fails to converge (F1=53.79%) — vanishing gradients in early layers confirm the necessity of adaptive learning rates for Transformers.

---

## 📚 Bibliography

- Vaswani et al. (2017). *Attention is All You Need.* NeurIPS.
- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.* arXiv:1810.04805.
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980.
- Loshchilov & Hutter (2017). *Decoupled Weight Decay Regularization.* arXiv:1711.05101.
- Shazeer & Stern (2018). *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.* arXiv:1804.04235.
- You et al. (2019). *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.* arXiv:1904.00962.
- Chen et al. (2023). *Lion: Evolved Sign Momentum.* arXiv:2302.06675.

---

## 👥 Authors

| Name | Program |
|---|---|
| Ghazli Mohamed Amine | Master DSBD & AI |
| Hamroudi Ayoub | Master DSBD & AI |
| Mouslim Jad | Master DSBD & AI |
| Boulatar Mouad | Master DSBD & AI |

**Academic Year** : 2025 - 2026  
**Module** : Optimization Algorithms  
**Submission Date** : June 20, 2026

---

## 📬 Repository

🔗 **https://github.com/BOULATARM/projet16-adaptive-gradient-methods-nlp**
