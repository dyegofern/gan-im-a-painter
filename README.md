# Mini Project (Week 5) - Introduction to Deep Learning

**Author:** Dyego Fernandes de Sousa
**Institution:** University of Colorado Boulder
**Course:** CSCA 5642 - Introduction to Deep Learning

---

## Project Overview

This project implements a **Generative Adversarial Network** for Monet painting style transfer, based on the Kaggle competition ["I'm Something of a Painter Myself"](https://www.kaggle.com/competitions/gan-getting-started). The goal is to transform regular photographs into images that mimic the artistic style of Claude Monet using Generative Adversarial Networks (GANs).

## Problem Description

The project showcases techniques learned during Week 5 of the course, specifically the application of Generative Adversarial Networks for image-to-image translation tasks. The implementation uses unpaired image-to-image translation, meaning the model learns to transfer style without requiring paired training examples.

**Dataset:**
- ~300 Monet paintings (256×256 RGB)
- ~7,000 photographs (256×256 RGB)
- Task type: Unpaired image-to-image translation

## Technical Approach

### Architecture

- **Generator:** CycleGAN architecture with residual blocks
- **Discriminator:** Pre-trained ResNet18 backbone (ImageNet weights) with custom classification head
- **Key Innovation:** Transfer learning using pre-trained weights significantly improved convergence speed

### Training Configuration

- **Epochs:** 200 (reduced to 30 due to computational constraints)
- **Batch Size:** 4
- **Learning Rate:** 0.0002 (with decay after epoch 100)
- **Loss Functions:**
  - Adversarial Loss (MSE)
  - Cycle Consistency Loss (L1, λ=10.0)
  - Identity Loss (L1, λ=5.0)
- **Optimizers:** Adam (β₁=0.5, β₂=0.999)

### Key Components

- **Replay Buffer:** Stores previously generated fake images to stabilize training
- **Learning Rate Scheduler:** Linear decay starting at epoch 100
- **Data Augmentation:** Custom transforms for training and testing

## Repository Structure

```
mod5/
├── monet_gan.ipynb           # Main training notebook
├── models.py                 # Generator and Discriminator architectures
├── utils.py                  # Dataset loader, transforms, and utilities
├── inspect_pretrained.py     # Model inspection and analysis tools
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory
│   ├── monet_jpg/           # Monet paintings
│   └── photo_jpg/           # Photographs
├── outputs/                  # Generated samples and plots
├── checkpoints/             # Saved model weights
└── submissions/             # Generated Monet-style images
```

## Results

The model successfully generates artistic transformations that capture Monet's impressionist style, including:
- Color palette transformation
- Brushstroke-like texture synthesis
- Atmospheric effects characteristic of impressionism

### Key Findings

- Pre-trained discriminator (ResNet18 from ImageNet) converges significantly faster than random initialization
- Transfer learning enables the discriminator to recognize real image features earlier in training
- Learning rate decay and replay buffer improve training stability

## Hyperparameter Experiments

Multiple configurations were tested:
- **Baseline:** λ_cycle=10.0, λ_identity=5.0
- **High Cycle Loss:** λ_cycle=20.0, λ_identity=5.0
- **High Identity Loss:** λ_cycle=10.0, λ_identity=10.0
- **Low Cycle Loss:** λ_cycle=5.0, λ_identity=5.0

## Installation & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch, torchvision, numpy, matplotlib, Pillow, tqdm, pandas, kaggle

### Training

Run the Jupyter notebook:
```bash
jupyter notebook monet_gan.ipynb
```

Or execute cells sequentially to:
1. Perform exploratory data analysis
2. Initialize models and training components
3. Train the CycleGAN
4. Generate Monet-style transformations
5. Visualize results and training curves

## Kaggle Competition

- **Notebook:** [deeplearning-gans](https://www.kaggle.com/code/dyegosousa/deeplearning-gans)
- Generated 7,038 Monet-style images for submission

## Future Improvements

- Experiment with StyleGAN2 architecture
- Implement attention mechanisms
- Enhanced data augmentation strategies
- Extended training duration for better convergence

## References

1. Zhu et al. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*
2. He et al. (2016). *Deep Residual Learning for Image Recognition*
3. Isola et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*
4. Kaggle Competition: [GAN Getting Started](https://www.kaggle.com/competitions/gan-getting-started)

---

**License:** Academic Use Only
**Contact:** University of Colorado Boulder
