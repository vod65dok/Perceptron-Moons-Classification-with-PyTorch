# Perceptron-Moons-Classification-with-PyTorch

# Overview

The project employs a simple Perceptron model to classify points from the "moons" dataset—a commonly used synthetic dataset for classification tasks. The dataset is split into training (70%) and testing (30%) subsets. The trained model is then evaluated on its ability to separate the two classes. A standout feature of this project is the graphical representation of the decision boundary, demonstrating the classifier's performance over the dataset.

# Features

Model Training: The Perceptron is implemented and trained using PyTorch, emphasizing simplicity and interpretability.
Custom Loss and Optimization: Training includes step-by-step calculations of loss and gradient updates using PyTorch operations.
Decision Boundary Visualization: Generates a visual plot showing the classifier's predictions across the feature space, making it easier to observe its behavior with non-linear data.
Dependencies

The following Python libraries are required to run this project:

- `torch`
- `matplotlib`
- `scikit-learn 
Install them via pip:
```bash
pip install torch matplotlib scikit-learn
