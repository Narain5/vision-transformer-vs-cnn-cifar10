# Vision Transformer vs CNN for CIFAR-10

## Overview
This project presents a comparative analysis of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for image classification on the CIFAR-10 dataset. Both models were implemented from scratch using PyTorch and evaluated using identical training pipelines.

## Dataset
- CIFAR-10 (50,000 training images, 10,000 test images)
- 10 classes including airplane, automobile, bird, cat, dog, ship, truck, etc.
- Data augmentation: random crop, horizontal flip
- Normalization using dataset-specific mean and std

## Models Implemented
### CNN Baseline
- 3 convolutional blocks with BatchNorm, MaxPooling, Dropout
- ~1.3M parameters
- Optimizer: Adam
- Test Accuracy: **61.24%**

### Vision Transformer (ViT)
- Patch embedding with Conv2D (patch size = 4)
- 9 Transformer encoder blocks
- Learnable class token + positional encoding
- Optimizer: AdamW with cosine annealing
- ~4M parameters
- Test Accuracy: **79.34%**

## Results Summary
| Model | Test Accuracy | Parameters |
|------|---------------|------------|
| CNN | 61.24% | ~1.3M |
| Vision Transformer | **79.34%** | ~4M |

Vision Transformer demonstrated superior generalization and class separability, even on a small-scale dataset like CIFAR-10.

## Visualizations
- Training & validation curves
- Confusion matrices
- Sample prediction outputs

## Deployment
The trained Vision Transformer model is exported in:
- PyTorch format (`.pth`)
- ONNX format (`.onnx`) for cross-platform inference

## Tech Stack
- Python
- PyTorch
- Torchvision
- NumPy, Matplotlib
- CUDA (GPU training)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Machine_learning_Final_project.ipynb
