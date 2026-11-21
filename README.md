# Neural Network using NumPy

A fully connected neural network built from scratch with **NumPy** for classifying handwritten digits from the **MNIST** dataset. The notebook implements the model mathematics, training loop, optimizers, regularization, evaluation, and visualization without using a deep-learning framework for the NumPy model.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soham20042005/Neural-Network-using-NumPY/blob/main/Neural_Network_using_NumPy.ipynb)

## Project contents

- [`Neural_Network_using_NumPy.ipynb`](Neural_Network_using_NumPy.ipynb) - complete, executable implementation and experiment.
- `README.md` - project documentation.

## What is included

- MNIST download and IDX-format data loading using Python's standard library
- Dense layers with configurable architectures
- ReLU, sigmoid, and tanh activation functions
- Numerically stable softmax and cross-entropy loss
- Manual forward propagation and backpropagation
- Mini-batch training with train/validation splits
- SGD, RMSProp, and Adam optimizers
- Optional batch normalization and dropout
- Finite-difference gradient checking
- Accuracy, loss curves, confusion matrix, and classification report
- Misclassified-example visualization
- Saving learned parameters to an `.npz` file
- Optional PyTorch benchmark for comparison only

## Model architecture

The default NumPy model is:

```text
784 inputs
  -> Dense(256)
  -> BatchNorm
  -> ReLU
  -> Dropout(0.2)
  -> Dense(128)
  -> BatchNorm
  -> ReLU
  -> Dropout(0.2)
  -> Dense(10)
  -> Softmax
```

The architecture, activation, optimizer, learning rate, batch normalization, dropout rate, batch size, and number of epochs can be changed in the notebook.

## Requirements

Core training uses:

- Python 3
- NumPy
- Matplotlib

The evaluation section also uses `scikit-learn`. PyTorch is optional and is only needed for the comparison section.

Install the dependencies with:

```bash
pip install numpy matplotlib scikit-learn
# Optional benchmark dependency:
# pip install torch
```

## Running the notebook

1. Open the notebook in [Google Colab](https://colab.research.google.com/github/soham20042005/Neural-Network-using-NumPY/blob/main/Neural_Network_using_NumPy.ipynb), or run it locally with Jupyter.
2. Execute the cells from top to bottom.
3. The notebook downloads MNIST into `data/mnist/` on the first run.
4. Review the gradient-check result before starting training.
5. Run the optional PyTorch section separately if PyTorch is installed.

A local Jupyter setup can be started with:

```bash
pip install notebook
jupyter notebook Neural_Network_using_NumPy.ipynb
```

Training is CPU-based for the NumPy implementation and may take several minutes depending on the machine.

## Reproducibility

The notebook uses a fixed random seed (`42`) for NumPy operations, data splitting, parameter initialization, and dropout. Runtime and final accuracy can still vary slightly across Python, NumPy, and hardware configurations.

## Learning goals

This project is intended as an educational implementation of neural-network fundamentals, including:

- How layers transform data during forward propagation
- How gradients are computed and propagated backward
- How optimizers update trainable parameters
- Why normalization and dropout help training
- How to validate gradients before training a larger model

## License

No license has been specified for this repository yet.
