from src.complex.layer import *
import numpy as np

class Network:
    """
    Represents a neural network composed of multiple, optionally different, layers. 
    The `Network` class manages the layers, performs forward and backward propagation, and updates the parameters of the layers during training.
    """

    def __init__(self, layers: list[Layer]) -> None:
        """
        Initializes the Network with a list of layers.
        
        Args:
            layers (list[Layer]): A list of Layer instances that make up the neural network.
        """
        self.layers = layers


    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Passes the input data through all layers of the network to compute the final output.
        
        Args:
            inputs (np.ndarray): The input data to the network.
        
        Returns:
            np.ndarray: The output of the network after processing the input through all layers.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    

    def backward(self, output_gradient: np.ndarray, eta: float) -> np.ndarray:
        """
        Performs backpropagation through all layers of the network to compute gradients and update parameters.
        
        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect to the output of the network.
            eta (float): The learning rate for parameter updates.
        """
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, eta)
        return output_gradient
    

    def train(self, X: np.ndarray, y: np.ndarray, loss_function: loss_func, epochs: int, eta: float, show_progress: bool=True) -> list[float]:
        """
        Trains the neural network on the provided dataset for a specified number of epochs.
        
        Args:
            X (np.ndarray): The input data for training.
            y (np.ndarray): The target labels for training.
            loss_function (loss_func): The loss function to use for training.
            epochs (int): The number of epochs to train the network.
            eta (float): The learning rate for parameter updates.
            show_progress (bool): Whether to display training progress.
        
        Returns:
            list[float]: A list of loss values for each epoch during training.
        """

        loss_fn = get_loss_function(loss_function)
        loss_dr = get_loss_derivative(loss_function)

        losses = []
        for epoch in range(epochs):
        
            output = self.forward(X)

            loss = loss_fn(output, y)
            losses.append(loss)

            output_gradient = loss_dr(output, y)
            self.backward(output_gradient, eta)

            if show_progress:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

        return losses
    

    def train_multi(self, X: list[np.ndarray] | np.ndarray, y: list[np.ndarray] | np.ndarray, loss_function: loss_func, generations: int, eta: float, num_epochs_per_generation: int, show_progress: bool=True) -> list[float]:
        """
        Trains the neural network on multiple input-target pairs for a specified number of epochs.
        
        Args:
            X (list[np.ndarray] | np.ndarray): A list of input data arrays or a 2D array of shape (n_samples, input_size) for training.
            y (list[np.ndarray] | np.ndarray): A list of target label arrays or a 2D array of shape (n_samples, output_size) for training.
            loss_function (loss_func): The loss function to use for training.
            generations (int): The number of generations to train the network.
            eta (float): The learning rate for parameter updates.
            num_epochs_per_generation (int): The number of epochs to train the network per generation.
            show_progress (bool): Whether to display training progress.
        
        Returns:
            list[float]: A list of average loss values for each generation during training.
        """

        losses = []

        if isinstance(X, np.ndarray) and X.ndim == 2:
            X_list = [X[i].reshape(1, -1) for i in range(X.shape[0])]
        else:
            X_list = X

        if isinstance(y, np.ndarray) and y.ndim == 2:
            y_list = [y[i].reshape(1, -1) for i in range(y.shape[0])]
        else:
            y_list = y

        for generation in range(generations):

            for X_i, y_i in zip(X_list, y_list):

                ep_losses = self.train(X_i, y_i, loss_function, epochs=num_epochs_per_generation, eta=eta, show_progress=show_progress)
                losses.append(np.mean(ep_losses))

            if show_progress:
                print(f'    [ Generation {generation + 1}/{generations}, Average Loss: {losses[-1]:.4f} ]')

        return losses
