# Machine Learning Algorithms from Scratch

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

A collection of fundamental machine learning algorithms implemented from scratch, focusing on the mathematical principles and core functionality behind popular ML techniques.

## Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Skills Demonstrated](#skills-demonstrated)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project contains implementations of essential machine learning algorithms built from the ground up. Rather than relying solely on high-level libraries, these implementations focus on the underlying mathematics and optimization techniques that power machine learning. The goal is to provide clear, educational implementations that demonstrate how these algorithms work internally.

## Algorithms Implemented

- **Linear Regression**
  - Univariate Linear Regression
  - Multivariate Linear Regression
  - Feature Scaling techniques
  
- **Logistic Regression**
  - Binary classification
  - Gradient descent optimization
  
- **Feature Engineering**
  - Z-score normalization
  - Feature scaling for multi-variable models

## Tech Stack

- **Python**: Core programming language
- **NumPy**: For numerical computations and matrix operations
- **Matplotlib**: For data visualization and plotting results
- **Jupyter Notebook**: Interactive development environment for algorithm implementation and testing
- **Scikit-learn**: For comparing custom implementations against standard library implementations

## Key Features

1. **Pure Python/NumPy Implementations**: Algorithms are built using only NumPy for mathematical operations, without relying on ML libraries for the core functionality
2. **Gradient Descent Optimization**: Custom implementation of gradient descent for both linear and logistic regression
3. **Feature Scaling Techniques**: Implementation of z-score normalization and other scaling approaches
4. **Visualization**: Comprehensive plotting of learning curves, decision boundaries, and model predictions
5. **Comparison with Scikit-learn**: Verification of custom implementations against industry-standard libraries

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Ml_algo.git
   cd Ml_algo
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install numpy matplotlib jupyter scikit-learn
   ```

## Usage Examples

### Running the Jupyter Notebooks

```bash
jupyter notebook
```

Then open any of the algorithm notebooks to see the implementations with explanations.

### Linear Regression Example

```python
# Load the univariate linear regression implementation
from univariate_linear_regression import compute_fwb, derivatives, compute_gd

# Sample training data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# Initialize parameters
w_init = 0.0
b_init = 0.0
alpha = 1.0e-2
iterations = 10000

# Run gradient descent
w, b, J_history = compute_gd(w_init, b_init, x_train, y_train, alpha, iterations)

# Make predictions
predictions = compute_fwb(x_train, w, b)
```

### Feature Scaling Example

```python
# Load the feature scaling implementation
from Feature_Scalling_Multivariable import zscore_norm

# Normalize the features
x_norm, x_mu, x_sigma = zscore_norm(x_train)

# Run gradient descent on normalized data
w_norm, b_norm, J_history = compute_gd(x_norm, y_train, w_i, b_i, 1000, 1.0e-1)
```

## Project Structure

- univariate_linear_regression.ipynb: Implementation of linear regression with one variable
- multiple_variable_linear_regression.ipynb: Implementation of linear regression with multiple variables
- Feature_Scalling_Multivariable.ipynb: Feature normalization techniques for multivariate regression
- Logistic_Regression_gradient_descent.ipynb: Implementation of logistic regression using gradient descent
- Linear_Regression_Sickit_Learn.ipynb: Comparison with Scikit-learn implementation
- data: Directory containing datasets used in the project
- kobeni.py: Helper functions used across multiple notebooks

## Skills Demonstrated

- **Mathematical Understanding**: Demonstrated through the implementation of gradient descent algorithms and feature scaling techniques based on mathematical principles
- **Algorithm Implementation**: Built complex algorithms from scratch, showing deep understanding of their internal workings
- **Data Manipulation**: Processed and transformed data using NumPy for efficient numerical operations
- **Data Visualization**: Created informative plots to visualize algorithm performance and results
- **Python Programming**: Utilized Python's scientific computing stack effectively for numerical algorithms
- **Machine Learning Concepts**: Applied fundamental ML concepts including cost functions, optimization, and model evaluation

## Future Improvements

- Implement additional algorithms (SVM, Decision Trees, Neural Networks)
- Add cross-validation techniques for model evaluation
- Create a unified API for all implemented algorithms
- Add more comprehensive documentation and mathematical explanations
- Implement regularization techniques to prevent overfitting

## License

MIT License

---

Created by [Your Name] | [Your Email] | [Your LinkedIn/GitHub]