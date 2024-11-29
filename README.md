# Perceptron-Moons-Classification-with-PyTorch
#Overview

This project implements a binary classification model using a Perceptron in PyTorch. It highlights the limitations of Perceptrons on non-linear datasets, such as the "moons" dataset, while visualizing the decision boundary to provide insights into model performance and training dynamics.

#Features

Model Training: Train a Perceptron using PyTorch on the "moons" dataset.
Custom Loss and Optimization: Implement custom loss computation and an optimization step.
Decision Boundary Visualization: Plot the decision boundary to illustrate the classifier's predictions across the input space.
Dataset and Workflow

The "moons" dataset, a popular synthetic dataset for classification tasks, is used for training and testing:

Training Set: 70% of the data is used for training the model.
Testing Set: 30% of the data is reserved for evaluation.
Visualization: The decision boundary is visualized to showcase the model's limitations on non-linear data.
Dependencies

Ensure the following Python libraries are installed:

torch (PyTorch)
matplotlib
scikit-learn
Install these dependencies using:

pip install torch matplotlib scikit-learn
This project serves as an educational tool for understanding Perceptrons and their challenges with non-linear datasets, with a practical demonstration of training and visualizing model performance.
