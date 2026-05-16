# Fashion MNIST Image Classification with CNN

A Deep Learning project that classifies clothing items from the **Fashion-MNIST** dataset using two CNN architectures trained with different optimizers and techniques.

---

## Problem Description

The goal of this project is to classify grayscale images of clothing items into **10 categories** using Convolutional Neural Networks (CNNs). We compare two model architectures to understand how design choices like optimizer selection, batch normalization, dropout, and learning rate scheduling affect performance.

**Classes:**
`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`

---

## Dataset

- **Name:** Fashion-MNIST
- **Source:** [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **Size:** 70,000 grayscale images (28×28 pixels)
  - Training set: 60,000 images
  - Test set: 10,000 images
- **Classes:** 10
- **Download:** Automatically downloaded via `torchvision.datasets.FashionMNIST`

---

## Model Architecture

### Model 1 — Basic CNN + SGD
| Layer | Details |
|-------|---------|
| Conv2d | 1 → 16 channels, kernel=3, padding=1 |
| ReLU | Activation |
| MaxPool2d | kernel=2, stride=2 |
| Flatten | — |
| Linear | 16×14×14 → 64 |
| ReLU | Activation |
| Linear | 64 → 10 |

- **Optimizer:** SGD (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 15

### Model 2 — Enhanced CNN + Adam
| Layer | Details |
|-------|---------|
| Conv2d | 1 → 32 channels, kernel=3, padding=1 |
| BatchNorm2d | 32 |
| ReLU | Activation |
| MaxPool2d | kernel=2, stride=2 |
| Conv2d | 32 → 64 channels, kernel=3, padding=1 |
| BatchNorm2d | 64 |
| ReLU | Activation |
| MaxPool2d | kernel=2, stride=2 |
| Flatten | — |
| Linear | 64×7×7 → 128 |
| ReLU | Activation |
| Dropout | p=0.5 |
| Linear | 128 → 10 |

- **Optimizer:** Adam (lr=0.01) + StepLR Scheduler (step=5, gamma=0.5)
- **Loss:** CrossEntropyLoss
- **Epochs:** 15

---

## Results

| Model | Optimizer | Enhancements | Final Val Accuracy | Final Val Loss |
|-------|-----------|-------------|-------------------|----------------|
| Model 1 (Basic CNN) | SGD | None | ~89% | ~0.28 |
| Model 2 (Enhanced CNN) | Adam + Scheduler | BatchNorm, Dropout, 2 Conv layers | ~92% | ~0.22 |

> **Note:** Exact values depend on the training run. Results are printed at the end of the notebook.

**Key Findings:**
- Model 2 significantly outperforms Model 1 due to deeper architecture and better regularization.
- Batch Normalization stabilizes and speeds up training.
- Adam with LR scheduling converges faster than plain SGD.

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision matplotlib pandas numpy
```

### Steps

1. **Install dependencies:**
   ```bash
   pip install torch torchvision matplotlib pandas numpy
   ```

2. **Run the notebook:**
   ```bash
   jupyter notebook project_Deeplearning.ipynb
   ```
   Or open it in **Google Colab** for free GPU access.

3. **Run all cells in order.** The dataset will be downloaded automatically on the first run.

---

## Requirements

```
torch
torchvision
matplotlib
pandas
numpy
jupyter
```

---
