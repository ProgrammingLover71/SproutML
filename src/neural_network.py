##============ Imports ============##


import numpy as np
from numpy.typing import ArrayLike
from .activations import *


##============ Internal Structures ============##


class WeightSet:

	"""
	A WeightSet holds the weights connecting a layer to the previous one, along with biases for the current layer.

	Weights are stored as a list of lists, where each inner list corresponds to the weights connecting one neuron in the current layer to all neurons in the previous layer. \n
	Biases are stored as a list, where each element corresponds to the bias for one neuron in the current layer.
	"""

	def __init__(self, num_current_layer: int, num_previous_layer: int):

		self.weights = np.random.uniform(-1, 1, (num_current_layer, num_previous_layer))
		self.biases  = np.random.uniform(-1, 1, num_current_layer)
	

	def get_weight_from_neuron_to_neuron(self, neuron_from: int, neuron_to: int) -> float:
		"""
		Returns the weight connecting a specific neuron in the previous layer to a specific neuron in the current layer.

		Args:
			neuron_from (int): The index of the neuron in the previous layer.
			neuron_to (int): The index of the neuron in the current layer.

		Returns:
			The scalar value of the weight connecting the specified neurons.
		"""

		return self.weights[neuron_to][neuron_from]


	def get_bias_for_neuron(self, neuron: int) -> float:
		"""
		Returns the bias for a specific neuron in the current layer.

		Args:
			neuron (int): The index of the neuron in the current layer.
		
		Returns:
			The scalar value of the bias for the specified neuron.
		"""

		return self.biases[neuron]


##============ Neural Network Class ============##


class NeuralNetwork:
	"""
	The NeuralNetwork class represents a simple feedforward neural network with customizable architecture, activation functions, and loss functions. 
	It supports forward propagation to compute outputs based on inputs and the defined weights and biases.
	"""

	def __init__(self, layer_structure: list[int], activation_function: activation_func = 'tanh', loss_function: loss_func = 'mse'):
		"""
		Initializes the neural network with the specified architecture, activation function, and loss function.

		Args:
			layer_structure (list[int]): A list of integers where each integer represents the number of neurons in that layer (including input and output layers).
			activation_function (activation_func): The name of the activation function to use ('sigmoid', 'relu', or 'tanh' - default 'tanh').
			loss_function (loss_func): The name of the loss function to use ('mse' for Mean Squared Error or 'cross_entropy' for Cross-Entropy Loss).
		"""

		self.layer_structure = layer_structure

		self.activation_function 	= get_activation_function(activation_function)
		self.activation_derivative 	= get_activation_derivative(activation_function)
		self.loss_function 			= get_loss_function(loss_function)
		
		self.weights = [WeightSet(layer_structure[i], layer_structure[i - 1]) for i in range(1, len(layer_structure))]
		
	#------------------------------------------------------------------------------------------------#
	
	def forward(self, inputs: np.ndarray | ArrayLike) -> np.ndarray:
		"""
		Performs forward propagation through the network to compute the output based on the given inputs.
		The `WeightSet`s are used to provide an easy calculation of the weighted sums, based on the following formula:

		```
		next_layer = activation(weights * current_layer + biases)
		```

		where:
		- `weights` is the matrix of weights connecting the current layer to the next layer.
		- `current_layer` is the input to the current layer.
		- `biases` is the vector of biases for the current layer.

		Args:
			inputs (np.ndarray | ArrayLike): A numpy array representing the input to the network (should match the size of the input layer).

		Returns:
			A numpy array representing the output of the network (size will match the output layer).
		"""

		# Normalize input to np.ndarray
		current_layer = np.asarray(inputs)

		for weight_set in self.weights:

			# Calculate the weighted sum for the next layer
			weighted_sum = np.dot(weight_set.weights, current_layer) + weight_set.biases

			# Apply the activation function
			current_layer = np.asarray(self.activation_function(weighted_sum))

		# At the end, current_layer will contain the values of the output layer
		return current_layer
	
	#------------------------------------------------------------------------------------------------#

	def backward(self, predictions: np.ndarray, targets: np.ndarray, learning_rate: float = 0.05) -> None:
		"""
		Performs backpropagation to compute gradients and update weights/biases.
		
		Args:
			predictions (np.ndarray): The output from forward pass.
			targets (np.ndarray): The expected output.
			learning_rate (float, optional - default 0.05): Controls the magnitude of weight updates.
		"""
		
		# Compute initial error (gradient of loss w.r.t. output)
		if self.loss_function == mean_squared_error:
			# For MSE: dL/dOutput = 2 * (predictions - targets) / n, where n is number of outputs
			error = 2 * (predictions - targets) / len(predictions)
		else:
			# For Cross-Entropy: dL/dOutput = predictions - targets
			error = predictions - targets
		
		# Work backward through each layer
		for i in range(len(self.weights) - 1, -1, -1):

			weight_set = self.weights[i]
			
			# Apply activation function derivative
			# For sigmoid: f'(x) = f(x) * (1 - f(x))
			# For ReLU: f'(x) = 1 if x > 0 else 0
			# For tanh: f'(x) = 1 - f(x)^2
			error *= self.activation_derivative(self.layer_preactivations[i])
			
			# Compute gradient w.r.t. weights: dL/dW = error * activation_input^T
			# activation_input is the output from the previous layer
			dW = np.outer(error, self.layer_activations[i])
			
			# Compute gradient w.r.t. biases: dL/db = error
			db = error
			
			# Compute error for the previous layer
			# error_prev = dL/d(pre-activation) @ weights = error @ weights^T
			error = np.dot(weight_set.weights.T, error)
			
			# Update weights and biases
			weight_set.weights -= learning_rate * dW
			weight_set.biases -= learning_rate * db

	#------------------------------------------------------------------------------------------------#

	def forward_with_cache(self, inputs: np.ndarray | ArrayLike) -> np.ndarray:
		"""
		Forward pass that caches intermediate values needed for backpropagation.
		"""
		inputs = np.asarray(inputs)

		self.layer_activations: list[np.ndarray]    = [inputs]  # Store activations
		self.layer_preactivations: list[np.ndarray] = []      	# Store pre-activation values
		
		current_layer = inputs
		
		for weight_set in self.weights:

			weighted_sum = np.dot(weight_set.weights, current_layer) + weight_set.biases
			self.layer_preactivations.append(weighted_sum)

			current_layer = np.asarray(self.activation_function(weighted_sum))
			self.layer_activations.append(current_layer)
		
		return current_layer
	

##============ Training Loop ============##


def train(net: NeuralNetwork, inputs: list[ArrayLike], targets: list[ArrayLike], epochs: int = 100, learning_rate: float = 0.05, show_msg: bool = False) -> np.ndarray:
	"""
	Trains a neural network using forward and backward propagation over multiple epochs.

	Args:
		inputs (list[ArrayLike]): A list of training input arrays, where each array represents a single sample.
		targets (list[ArrayLike]): A list of target output arrays, where each array represents the expected output for the corresponding input sample.
		epochs (int, optional - default 100): Number of training iterations.
		learning_rate (float, optional - default 0.05): Step size for weight updates during backpropagation.
		show_msg (bool, optional - default False): Whether to show the best loss and epoch every 100 epochs.

	Returns:
		A `numpy.ndarray` containing the average losses recorded at each epoch, which can be used for monitoring training progress.

	Notes:
		The method performs the following steps for each epoch:
		1. Forward pass: Computes predictions and caches intermediate values for backpropagation
		2. Loss calculation: Evaluates the difference between predictions and targets for each sample
		3. Backward pass: Computes gradients and updates network weights using the learning rate
		The loss is accumulated across all samples in an epoch and then averaged.
	"""

	new_inputs  = np.asarray(inputs)
	new_targets = np.asarray(targets)

	losses = np.zeros(epochs)
	num_samples = len(inputs)

	for epoch in range(epochs):
		epoch_loss = 0.0

		for X, y in zip(new_inputs, new_targets):
				
			# Forward pass (with caching)
			predictions = net.forward_with_cache(X)
				
			# Calculate loss for this sample
			loss = net.loss_function(predictions, y)
			epoch_loss += loss
				
			# Backward pass
			net.backward(predictions, y, learning_rate)
			
		# Average the loss for this epoch
		losses[epoch] = epoch_loss / num_samples

		if epoch % 100 == 0 and show_msg:
			print(f"Epoch #{epoch + 1} Best loss: {losses[epoch]}")
		
	return losses
