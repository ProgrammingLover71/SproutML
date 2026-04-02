##============ Imports ============##


from src.complex.layer import Layer
from src.activations import *

import numpy as np


##============ Recurrent Layer Class ============##


class RecurrentLayer(Layer):
	"""
	Represents a recurrent neural network (RNN) layer.
	This layer processes sequences of data, maintaining a hidden state that captures information from previous time steps.
	"""

	def __init__(self, input_size: int, hidden_size: int, output_size: int | None = None, activation: activation_func = 'tanh'):
		super().__init__()

		self.input_size  = input_size
		self.hidden_size = hidden_size
		self.output_size = hidden_size if output_size is None else output_size

		self.output_activation_name       = activation
		self.output_activation            = get_activation_function(activation)
		self.output_activation_derivative = get_activation_derivative(activation)

		self.W_xh = np.random.uniform(-1, 1, (input_size, hidden_size))
		self.W_hh = np.random.uniform(-1, 1, (hidden_size, hidden_size))
		self.W_hy = np.random.uniform(-1, 1, (hidden_size, self.output_size))

		self.b_h = np.zeros((1, hidden_size))
		self.b_y = np.zeros((1, self.output_size))

		self.hidden_state = np.zeros((1, hidden_size))

		self.inputs_cache = []
		self.h_cache      = []
		self.a_cache      = []
		self.y_cache      = []


	def forward(self, inputs: np.ndarray, hidden_state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
		"""
		Forward pass for recurrent layer over a batch/time sequence.

		Args:
			inputs (np.ndarray): Input sequence data.
			hidden_state (np.ndarray): The current hidden state for recurrent layers.

		Returns:
			np.ndarray: Output sequence data.
			np.ndarray: Updated hidden state.
		"""
		x = np.array(inputs)
		if x.ndim == 1:
			x = x.reshape(1, -1)

		if x.ndim != 2 or x.shape[1] != self.input_size:
			raise ValueError(f"RecurrentLayer expects input shape (seq_len, {self.input_size}), got {x.shape}")

		h = self.hidden_state.copy() if hidden_state is None else np.array(hidden_state)
		if h.ndim == 1:
			h = h.reshape(1, -1)

		if h.shape != (1, self.hidden_size):
			raise ValueError(f"hidden_state must have shape (1, {self.hidden_size}), got {h.shape}")

		self.inputs_cache = []
		self.h_cache      = [h.copy()]
		self.a_cache      = []
		self.y_cache      = []

		outputs = []

		for t in range(x.shape[0]):
			x_t = x[t:t+1]
			a_t = x_t @ self.W_xh + h @ self.W_hh + self.b_h
			h = np.tanh(a_t)

			y = self.output_activation(h @ self.W_hy + self.b_y)

			self.inputs_cache.append(x_t)
			self.a_cache.append(a_t)
			self.h_cache.append(h.copy())
			self.y_cache.append(y.copy())	# type: ignore -- y will be np.ndarray, but the activation function can return float or np.ndarray depending on the input
			outputs.append(y)

		self.hidden_state = h.copy()

		return np.vstack(outputs), h


	def backward(self, output_gradient: np.ndarray, eta: float, hidden_gradient: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
		"""
		Backward pass for recurrent layer through time (BPTT).

		Args:
			output_gradient (np.ndarray): Gradient w.r.t. the output.
			eta (float): Learning rate.
			hidden_gradient (np.ndarray | None): The gradient of the loss with respect to the hidden state.

		Returns:
			tuple[np.ndarray, np.ndarray]: (dx, dh) gradient w.r.t inputs and initial hidden state.
		"""

		grad = np.array(output_gradient)
		if grad.ndim == 1:
			grad = grad.reshape(1, -1)

		seq_len = len(self.inputs_cache)
		if seq_len == 0:
			raise ValueError("No cached forward pass data for backward pass")

		if grad.shape[0] != seq_len:

			if grad.shape[0] == 1:
				grad = np.repeat(grad, seq_len, axis=0)

			else:
				raise ValueError(f"output_gradient.shape {grad.shape} does not match sequence length {seq_len}")

		dh_next = np.zeros((1, self.hidden_size)) if hidden_gradient is None else np.array(hidden_gradient)
		if dh_next.ndim == 1:
			dh_next = dh_next.reshape(1, -1)

		dW_xh = np.zeros_like(self.W_xh)
		dW_hh = np.zeros_like(self.W_hh)
		dW_hy = np.zeros_like(self.W_hy)

		db_h = np.zeros_like(self.b_h)
		db_y = np.zeros_like(self.b_y)
		dx   = np.zeros((seq_len, self.input_size))

		for t in reversed(range(seq_len)):
			x_t = self.inputs_cache[t]
			h_prev = self.h_cache[t]
			h_curr = self.h_cache[t + 1]

			y_curr = self.y_cache[t]
			out_deriv = self.output_activation_derivative(h_curr @ self.W_hy + self.b_y)

			dy = grad[t:t+1] * out_deriv

			dW_hy += h_curr.T @ dy
			db_y += np.sum(dy, axis=0, keepdims=True)

			dh_from_y = dy @ self.W_hy.T

			dh = dh_from_y + dh_next
			da = dh * (1 - np.tanh(self.a_cache[t]) ** 2)

			dW_xh += x_t.T @ da
			dW_hh += h_prev.T @ da
			db_h += da

			dx[t:t+1] = da @ self.W_xh.T
			dh_next = da @ self.W_hh.T

		self.W_xh -= eta * dW_xh
		self.W_hh -= eta * dW_hh
		self.W_hy -= eta * dW_hy
		self.b_h  -= eta * db_h
		self.b_y  -= eta * db_y

		return dx, dh_next
