##============ Imports ============##


from src.activations import *

import numpy as np


##============ Layer Class ============##


class Layer:
    """
    Represents the base class for a layer in a neural network. It is extended to create certain types of layers, such as dense layers, convolutional layers, etc. 
    The Layer class provides a common interface for all types of layers, including methods for forward and backward propagation, as well as parameter updates.
    """

    def __init__(self) -> None:
        pass
        

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Passes the input data through the layer and computes the output. This method should be overridden by subclasses to implement specific layer behavior.
        
        Args:
            inputs (np.ndarray): The input data to the layer.
        
        Returns:
            np.ndarray: The output of the layer after processing the input.
        """
        return inputs


    def backward(self, output_gradient: np.ndarray, eta: float) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the inputs of the layer, given the gradient of the loss with respect to the outputs.
        
        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect to the outputs of the layer.
            eta (float): The learning rate for parameter updates.
        
        Returns:
            np.ndarray: The gradient of the loss with respect to the inputs of the layer.
        """
        return output_gradient


class RecurrentLayer(Layer):
    """
    Represents a recurrent neural network (RNN) layer.
    This is a stub implementation used as a placeholder in the architecture.
    Subclasses should implement specific recurrent cell behavior (e.g. SimpleRNN, LSTM, GRU).
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_state = np.zeros((1, hidden_size))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass for recurrent layer over a batch/time sequence.
        This base stub does not implement recurrence and must be overridden.

        Args:
            inputs (np.ndarray): Input sequence data.

        Returns:
            np.ndarray: Output sequence data.
        """
        raise NotImplementedError("RecurrentLayer.forward() should be implemented by a subclass")

    def backward(self, output_gradient: np.ndarray, eta: float) -> np.ndarray:
        """
        Backward pass for recurrent layer through time (BPTT).
        This base stub does not implement recurrence and must be overridden.

        Args:
            output_gradient (np.ndarray): Gradient w.r.t. the output.
            eta (float): Learning rate.

        Returns:
            np.ndarray: Gradient w.r.t. the input.
        """
        raise NotImplementedError("RecurrentLayer.backward() should be implemented by a subclass")


##============ Layer subclasses ============##


# ------ Dense Layer ------#

class DenseLayer(Layer):
    """
    Represents a fully connected (dense) layer in a neural network. This layer performs a linear transformation of the input data, followed by an optional activation function.
    """

    def __init__(self, input_size: int, output_size: int, activation: activation_func = 'relu') -> None:
        """
        Initializes the DenseLayer with the specified input size, output size, and activation function.
        
        Args:
            input_size (int): The number of input neurons.
            output_size (int): The number of output neurons.
            activation (activation_func): The activation function to apply after the linear transformation.
        """
        super().__init__()

        self.input_size  = input_size
        self.output_size = output_size

        self.activation_name        = activation
        self.activation             = get_activation_function(activation)
        self.activation_derivative  = get_activation_derivative(activation)

        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.biases  = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the dense layer, applying the linear transformation and activation function.
        
        Args:
            inputs (np.ndarray): The input data to the layer.
        
        Returns:
            np.ndarray: The output of the layer after processing the input.
        """

        self.inputs = inputs

        self.linear_output    = inputs @ self.weights + self.biases
        self.activated_output = self.activation(self.linear_output)

        return self.activated_output    # type: ignore -- the return type will be np.ndarray, but the activation returns float or np.ndarray depending on the input

    def backward(self, output_gradient: np.ndarray, eta: float) -> np.ndarray:
        """
        Computes the backward pass through the dense layer, calculating the gradients and updating the weights and biases.
        
        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect to the outputs of the layer.
            eta (float): The learning rate for parameter updates.

        Returns:
            np.ndarray: The gradient of the loss with respect to the inputs of the layer.
        """

        transposed_weights = self.weights.T     # type: np.ndarray
        transposed_inputs  = self.inputs.T      # type: np.ndarray

        activation_gradient = self.activation_derivative(self.linear_output) * output_gradient  # type: np.ndarray
        input_gradient      = activation_gradient @ transposed_weights                   # type: np.ndarray

        weights_gradient    = transposed_inputs @ activation_gradient                    # type: np.ndarray
        biases_gradient     = np.sum(activation_gradient, axis=0, keepdims=True)                # type: np.ndarray

        # Update weights and biases
        self.weights -= eta * weights_gradient
        self.biases  -= eta * biases_gradient

        return input_gradient


# ------ GRU Layer (imported) ------#

from src.complex.gru import GRULayer    # Import it here to avoid circular import


# ------ Sigmoid Layer (Yes / No) ------#

class Sigmoid(Layer):
    """
    Represents a sigmoid activation layer in a neural network. This layer applies the sigmoid activation function to the input data.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid activation function to the input data.
        
        Args:
            inputs (np.ndarray): The input data to the layer.
        
        Returns:
            np.ndarray: The output of the layer after applying the sigmoid activation function.
        """
        self.inputs = inputs
        return 1 / (1 + np.exp(-inputs))


    def backward(self, output_gradient: np.ndarray, eta: float) -> np.ndarray:
        """
        Computes the backward pass through the sigmoid layer, calculating the gradient of the loss with respect to the inputs.
        
        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect to the outputs of the layer.
            eta (float): The learning rate for parameter updates (not used in this layer).
        
        Returns:
            np.ndarray: The gradient of the loss with respect to the inputs of the layer.
        """
        sigmoid_output = self.forward(self.inputs)  # type: np.ndarray
        return output_gradient * sigmoid_output * (1 - sigmoid_output)  # type: np.ndarray


# ------ Softmax Layer (Classification) ------#

class Softmax(Layer):
    """
    Represents a softmax activation layer in a neural network. 
    This layer applies the softmax activation function to the input data, typically used for multi-class classification problems.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Applies the softmax activation function to the input data.
        
        Args:
            inputs (np.ndarray): The input data to the layer.
        
        Returns:
            np.ndarray: The output of the layer after applying the softmax activation function.
        """
        self.inputs = inputs

        exp_values    = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # type: np.ndarray
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # type: np.ndarray

        return probabilities
    

    def backward(self, output_gradient: np.ndarray, eta: float) -> np.ndarray:
        """
        Computes the backward pass through the softmax layer, calculating the gradient of the loss with respect to the inputs.
        
        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect to the outputs of the layer.
            eta (float): The learning rate for parameter updates (not used in this layer).
        
        Returns:
            np.ndarray: The gradient of the loss with respect to the inputs of the layer.
        """
        softmax_output = self.forward(self.inputs)  # type: np.ndarray
        return output_gradient * softmax_output * (1 - softmax_output)  # type: np.ndarray
