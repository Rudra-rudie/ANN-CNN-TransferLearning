# Fashion-MNIST Image Classification: ANN vs CNN vs Transfer Learning

A comparative study of three deep learning approaches for classifying clothing images from the Fashion-MNIST dataset, with automated hyperparameter tuning via Optuna.

## 🎯 Overview

This project explores how model architecture choice impacts performance on an image classification task. Three approaches were implemented and benchmarked against each other:

1. **Artificial Neural Network (ANN)** — a fully connected baseline
2. **Convolutional Neural Network (CNN)** — purpose-built for spatial image data
3. **Transfer Learning (VGG16)** — leveraging a pretrained ImageNet model

## 📊 Results

| Model | Accuracy | Tuning Method |
|---|---|---|
| ANN (baseline) | ~88.8% | Optuna (learning rate, optimizer, dropout, hidden layer sizes) |
| CNN | ~90.65% | Optuna (learning rate, dropout rates) |
| Transfer Learning (VGG16) | Implemented for comparison | Pretrained feature extraction |

**Key takeaway:** CNNs outperformed the fully connected ANN by ~2 percentage points, confirming that convolutional layers capture spatial patterns in image data more effectively than flattened pixel inputs alone.

## 🛠️ Tech Stack

- **PyTorch** — model building and training
- **torchvision** — VGG16 pretrained model and image transforms
- **Optuna** — automated hyperparameter optimization
- **pandas / scikit-learn** — data loading and train/test splitting
- **matplotlib** — visualization

## 🔍 Process

1. **Data preparation** — Loaded the Fashion-MNIST dataset (28x28 grayscale clothing images across 10 categories), split into train/test sets.
2. **Data quality check** — Identified and fixed NaN values in the input features that were causing unstable (NaN) training loss.
3. **ANN baseline** — Built a configurable fully connected network and tuned hidden layer count, neuron count, learning rate, optimizer, and dropout using Optuna (15+ trials).
4. **CNN** — Designed a 3-layer convolutional architecture with batch normalization and dropout, tuned via Optuna for learning rate and dropout rates.
5. **Transfer learning** — Applied a pretrained VGG16 model with custom image transforms (resize, center crop) for feature extraction.
6. **Evaluation** — Compared final test-set accuracy across all three approaches.

## 📁 Repository Structure

```
├── Fashion_MNIST_ANN_CNN_TransferLearning.ipynb   # Full notebook
└── README.md
```

## 🚀 How to Run

1. Clone this repo
2. Open the notebook in Google Colab or Jupyter (GPU recommended)
3. Download the [Fashion-MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and update the file path
4. Run cells sequentially

## 💡 Next Steps

- Push CNN accuracy further with data augmentation (rotation, flipping, scaling)
- Fine-tune VGG16 layers rather than using it purely for feature extraction
- Try more recent lightweight architectures (e.g. EfficientNet) for comparison

---

*Built as a hands-on exploration of neural network architectures for image classification.*
