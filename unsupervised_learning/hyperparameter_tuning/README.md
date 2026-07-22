# Hyperparameter Tuning

This project covers the implementation of techniques used for hyperparameter optimization in machine learning.

The main focus of the project is Bayesian Optimization using Gaussian Processes. The implementations include covariance kernels, Gaussian Process prediction, acquisition functions, and the optimization of black-box functions.

## Learning Objectives

By the end of this project, I should be able to explain:

* What hyperparameters are
* The difference between model parameters and hyperparameters
* What hyperparameter tuning is
* Common approaches to hyperparameter optimization
* What Grid Search is
* What Random Search is
* What Bayesian Optimization is
* What a Gaussian Process is
* How Gaussian Processes can be used for optimization
* What covariance kernels are
* How the Radial Basis Function kernel works
* What acquisition functions are
* How Expected Improvement is calculated
* How Bayesian Optimization selects the next point to evaluate

## Requirements

### General

* All files are interpreted or compiled on Ubuntu
* All files are written in Python 3
* All files end with a new line
* The first line of every Python file is:

```python
#!/usr/bin/env python3
```

* Code follows the `pycodestyle` style guide
* All modules have documentation
* All classes have documentation
* All functions and methods have documentation
* NumPy is used for numerical operations

## Project Structure

| File        | Description                                                               |
| ----------- | ------------------------------------------------------------------------- |
| `0-gp.py`   | Defines a class representing a noiseless one-dimensional Gaussian Process |
| `0-main.py` | Tests the initialization and RBF kernel of the Gaussian Process           |
| `README.md` | Contains information about the project                                    |

## Tasks

### 0. Initialize Gaussian Process

The file `0-gp.py` defines the class:

```python
class GaussianProcess:
```

The class represents a noiseless one-dimensional Gaussian Process.

The constructor has the following form:

```python
def __init__(self, X_init, Y_init, l=1, sigma_f=1):
```

The class stores:

* `X`: inputs already evaluated by the black-box function
* `Y`: outputs produced by the black-box function
* `l`: length parameter of the kernel
* `sigma_f`: standard deviation of the black-box function output
* `K`: covariance kernel matrix of the existing samples

The class also contains the method:

```python
def kernel(self, X1, X2):
```

This method calculates the covariance matrix between two collections of input points using the Radial Basis Function kernel.

## Radial Basis Function Kernel

The RBF kernel is defined as:

[
k(x_i, x_j) =
\sigma_f^2
\exp\left(
-\frac{(x_i-x_j)^2}{2l^2}
\right)
]

Where:

* (x_i) and (x_j) are input points
* (l) is the length parameter
* (\sigma_f) controls the vertical scale of the function
* (k(x_i, x_j)) represents the covariance between the two points

Points that are close together have a high covariance, while points that are far apart have a lower covariance.

For a point compared with itself:

[
k(x_i, x_i) = \sigma_f^2
]

because the squared distance is zero.

## Usage

Run the main test file with:

```bash
./0-main.py
```

Example output:

```text
True
True
0.6
2
(2, 2) [[4.         0.13150595]
 [0.13150595 4.        ]]
True
```

## Repository

This project is part of the Machine Learning curriculum and focuses on implementing hyperparameter optimization algorithms from scratch.

