# Vision Transformer vs CNN for CIFAR-10

## Overview
This project presents a comparative study of **Convolutional Neural Networks (CNNs)** and **Vision Transformer (ViT)** architectures for image classification on the **CIFAR-10** dataset. Both models were implemented, trained, and evaluated using **PyTorch**, with a focus on performance, generalization, and class-level behavior.

The results demonstrate that Vision Transformers can significantly outperform traditional CNNs even on relatively small image datasets when trained with appropriate optimization and regularization strategies.

---

## Dataset
- **CIFAR-10**
- 60,000 RGB images of size **32×32**
  - 50,000 training images
  - 10,000 test images
- 10 classes:
  - airplane, automobile, bird, cat, deer  
  - dog, frog, horse, ship, truck
- Preprocessing and augmentation:
  - Normalization using dataset-specific mean and standard deviation
  - Random horizontal flip
  - Random cropping

---

## Models Implemented

### CNN Baseline
- 3 convolutional blocks:
  - Conv2D → Batch Normalization → ReLU
  - MaxPooling and Dropout
- Fully connected classification head
- Optimizer: **Adam**
- Number of parameters: ~1.3M

### Vision Transformer (ViT)
- Patch embedding using Conv2D (patch size = 4)
- Learnable class token and positional embeddings
- 9 Transformer encoder blocks:
  - Multi-head self-attention
  - MLP with residual connections
- Optimizer: **AdamW**
- Learning rate scheduler: **Cosine Annealing**
- Number of parameters: ~4M

---

## Training Pipeline
- Loss Function: **Cross Entropy Loss**
- Batch Size: **128**
- Epochs: **40**
- Regularization:
  - Dropout
  - Weight decay (ViT)
- Training performed on **GPU (CUDA)**
- Evaluation metrics:
  - Accuracy
  - Confusion matrices

---

## Results

| Model | Test Accuracy |
|------|---------------|
| CNN | 61.24% |
| Vision Transformer | **79.34%** |

### Key Observations
- Vision Transformer achieves superior generalization compared to CNN
- Reduced confusion among visually similar classes
- Stable training behavior without severe overfitting

---

## Visualizations
The project includes:
- Confusion matrices for CNN and Vision Transformer
- Sample prediction comparisons
- Class-wise prediction confidence analysis

All prediction-related visual outputs are available in the **`Predictions/`** directory.

---

## Repository Structure
```text
vision-transformer-vs-cnn-cifar10/
├── Notebooks/
│   └── Vision_transformer_CIFAR.ipynb
├── Predictions/
│   ├── CNN_Predictions.png
│   └── Vision_transformer_Predictions.png
├── Report/
│   └── Project_Report.pdf
└── README.md
