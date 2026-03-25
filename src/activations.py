##============ Imports ============##


import numpy as np
from typing import Literal


##============ Value Types ============##


activation_func = Literal['sigmoid', 'relu', 'silu', 'tanh']
loss_func 		= Literal['mse', 'cross_entropy']
scalar			= int | float


def get_activation_function(name: activation_func):
	if name == 'sigmoid':
		return sigmoid
	elif name == 'relu':
		return relu
	elif name == 'tanh':
		return tanh
	elif name == 'silu':
		return silu
	else:
		raise ValueError(f"Unsupported activation function: {name}")
	

def get_activation_derivative(name: activation_func):
	if name == 'sigmoid':
		return sigmoid_derivative
	elif name == 'relu':
		return relu_derivative
	elif name == 'tanh':
		return tanh_derivative
	elif name == 'silu':
		return silu_derivative
	else:
		raise ValueError(f"Unsupported activation function: {name}")


def get_loss_function(name: loss_func):
	if name == 'mse':
		return mean_squared_error
	elif name == 'cross_entropy':
		return cross_entropy_loss
	else:
		raise ValueError(f"Unsupported loss function: {name}")
		


##============ Activation Functions & Derivatives ============##


#------ Sigmoid ------#

def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
	"""The sigmoid activation function maps any real-valued input to a value between 0 and 1, following an S-shaped curve."""
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: float | np.ndarray) -> float | np.ndarray:
	"""Computes the derivative of the sigmoid function."""
	s = sigmoid(x)
	return s * (1 - s)

#------ Rectified Linear Unit (ReLU) ------#

def relu(x: float | np.ndarray) -> float | np.ndarray:
	"""The ReLU (Rectified Linear Unit) activation function outputs the input directly if it is positive; otherwise, it outputs zero."""
	return np.maximum(0, x)

def relu_derivative(x: float | np.ndarray) -> float | np.ndarray:
	"""Computes the derivative of the ReLU function."""
	return np.where(x > 0, 1, 0)

#------ Hyperbolic Tangent (tanh) ------#

def tanh(x: float | np.ndarray) -> float | np.ndarray:
	"""The tanh (hyperbolic tangent) activation function maps any real-valued input to a value between -1 and 1, following an S-shaped curve."""
	return np.tanh(x)

def tanh_derivative(x: float | np.ndarray) -> float | np.ndarray:
	"""Computes the derivative of the tanh function."""
	return 1 - np.tanh(x) ** 2

#------ Sigmoid Linear Unit (SiLU) ------#

def silu(x: float | np.ndarray) -> float | np.ndarray:
	"""The Sigmoid Linear Unit (SiLU) is a smoothed version of ReLU, using the sigmoid function as its base."""
	return x * sigmoid(x)

def silu_derivative(x: float | np.ndarray) -> float | np.ndarray:
	"""Computes the derivative of the SiLU function."""
	A = x + np.sinh(x)
	B = 4 * np.cosh(x / 2.0) ** 2
	return A / B + 0.5


##============ Loss Functions ============##



def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
	"""The Mean Squared Error (MSE) loss function calculates the average of the squares of the differences between predicted and actual values."""

	return float(np.mean((predictions - targets) ** 2))


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
	"""The Cross-Entropy Loss function measures the performance of a classification model whose output is a probability value between 0 and 1."""

	epsilon = 1e-15  # To prevent log(0)
	predictions = np.clip(predictions, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
	
	return float(-np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)))
