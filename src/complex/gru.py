##============ Imports ============##


from src.complex.layer import RecurrentLayer

import numpy as np


##============ GRU Layer ============##


class GRULayer(RecurrentLayer):
    """
    Represents a Gated Recurrent Unit (GRU) layer in a neural network. This layer is designed to capture temporal dependencies in sequential data.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(input_size, hidden_size)

        self.W_z = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.U_z = np.random.uniform(-1, 1, (hidden_size, hidden_size))
        self.b_z = np.zeros((1, hidden_size))

        self.W_r = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.U_r = np.random.uniform(-1, 1, (hidden_size, hidden_size))
        self.b_r = np.zeros((1, hidden_size))

        self.W_h = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.U_h = np.random.uniform(-1, 1, (hidden_size, hidden_size))
        self.b_h = np.zeros((1, hidden_size))

        self.inputs_cache = []
        self.h_cache = []
        self.z_cache = []
        self.r_cache = []

        self.h_tilde_cache = []
        self.a_z_cache = []
        self.a_r_cache = []
        self.a_h_cache = []

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass for GRU layer over a time sequence.

        Args:
            inputs (np.ndarray): Input sequence of shape (seq_len, input_size) or (input_size,).

        Returns:
            np.ndarray: Output hidden states of shape (seq_len, hidden_size).
        """
        x = np.array(inputs)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"GRULayer expects input shape (seq_len, {self.input_size}), got {x.shape}")

        self.inputs_cache = []
        self.h_cache = [self.hidden_state.copy()]
        self.z_cache = []
        self.r_cache = []

        self.h_tilde_cache = []
        self.a_z_cache = []
        self.a_r_cache = []
        self.a_h_cache = []

        h = self.hidden_state.copy()
        outputs: list[np.ndarray] = []

        for t in range(x.shape[0]):

            x_t = x[t:t+1]
            a_z = x_t @ self.W_z + h @ self.U_z + self.b_z

            z = 1 / (1 + np.exp(-a_z))
            a_r = x_t @ self.W_r + h @ self.U_r + self.b_r

            r = 1 / (1 + np.exp(-a_r))
            a_h = x_t @ self.W_h + (r * h) @ self.U_h + self.b_h

            h_tilde = np.tanh(a_h)

            h: np.ndarray = (1 - z) * h + z * h_tilde

            self.inputs_cache.append(x_t)
            self.h_cache.append(h.copy())
            self.z_cache.append(z)
            self.r_cache.append(r)

            self.h_tilde_cache.append(h_tilde)
            self.a_z_cache.append(a_z)
            self.a_r_cache.append(a_r)
            self.a_h_cache.append(a_h)

            outputs.append(h)

        self.hidden_state = h.copy()

        return np.vstack(outputs)


    def backward(self, output_gradient: np.ndarray, eta: float) -> np.ndarray:
        """
        Backpropagation through time for GRU layer.

        Args:
            output_gradient (np.ndarray): Gradient wrt outputs, shape (seq_len, hidden_size) or (hidden_size,).
            eta (float): Learning rate.

        Returns:
            np.ndarray: Gradient wrt inputs, shape (seq_len, input_size).
        """
        dW_z = np.zeros_like(self.W_z)
        dU_z = np.zeros_like(self.U_z)
        db_z = np.zeros_like(self.b_z)

        dW_r = np.zeros_like(self.W_r)
        dU_r = np.zeros_like(self.U_r)
        db_r = np.zeros_like(self.b_r)

        dW_h = np.zeros_like(self.W_h)
        dU_h = np.zeros_like(self.U_h)
        db_h = np.zeros_like(self.b_h)

        x = np.vstack(self.inputs_cache)
        seq_len = x.shape[0]

        dh_next = np.zeros((1, self.hidden_size))
        dx = np.zeros((seq_len, self.input_size))

        grad = np.array(output_gradient)
        if grad.ndim == 1:
            grad = grad.reshape(1, -1)

        if grad.shape[0] != seq_len:

            if grad.shape[0] == 1:
                grad = np.repeat(grad, seq_len, axis=0)
            else:
                raise ValueError(f"output_gradient.shape {grad.shape} does not match sequence length {seq_len}")

        for t in reversed(range(seq_len)):

            x_t = x[t:t+1]
            h_prev = self.h_cache[t]
            h = self.h_cache[t + 1]
            z = self.z_cache[t]
            r = self.r_cache[t]

            h_tilde = self.h_tilde_cache[t]
            a_z = self.a_z_cache[t]
            a_r = self.a_r_cache[t]
            a_h = self.a_h_cache[t]

            dh = grad[t:t+1] + dh_next

            dh_tilde = dh * z
            da_h = dh_tilde * (1 - np.tanh(a_h) ** 2)

            dz = dh * (h_tilde - h_prev)
            da_z = dz * z * (1 - z)

            dr = da_h @ self.U_h.T * h_prev
            da_r = dr * r * (1 - r)

            dW_h += x_t.T @ da_h
            dU_h += (r * h_prev).T @ da_h
            db_h += da_h

            dW_z += x_t.T @ da_z
            dU_z += h_prev.T @ da_z
            db_z += da_z

            dW_r += x_t.T @ da_r
            dU_r += h_prev.T @ da_r
            db_r += da_r

            dx[t] = da_h @ self.W_h.T + da_z @ self.W_z.T + da_r @ self.W_r.T

            dh_prev = dh * (1 - z)
            dh_prev += da_h @ self.U_h.T * r
            dh_prev += da_z @ self.U_z.T
            dh_prev += da_r @ self.U_r.T

            dh_next = dh_prev

        self.W_h -= eta * dW_h
        self.U_h -= eta * dU_h
        self.b_h -= eta * db_h

        self.W_z -= eta * dW_z
        self.U_z -= eta * dU_z
        self.b_z -= eta * db_z

        self.W_r -= eta * dW_r
        self.U_r -= eta * dU_r
        self.b_r -= eta * db_r

        return dx
