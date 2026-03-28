import src.complex.network as network

def test_network():

    layers: list[network.Layer] = [
        network.DenseLayer(input_size=2, output_size=4, activation='relu'),
        network.DenseLayer(input_size=4, output_size=4, activation='sigmoid'),
        network.DenseLayer(input_size=4, output_size=2, activation='sigmoid'),
        network.Softmax()
    ]

    net = network.Network(layers)

    print("Testing forward pass...")
    inputs = network.np.array([0, 1])
    output = net.forward(inputs)

    print("Output:", output)

    print("\nTesting backward pass...")
    X0 = network.np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y0 = network.np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # XOR targets  (first is class 0, second is class 1)

    net.train_multi(X0, y0, loss_function = 'cross_entropy', generations = 100, eta = 0.1, num_epochs_per_generation = 250, show_progress = True)
    
    print("\nTesting trained network...")
    for X, y in zip(X0, y0):
        X = X.reshape(1, -1)    # type: network.np.ndarray
        y = y.reshape(1, -1)    # type: network.np.ndarray
        output = net.forward(X)
        print(f"Input: {X.flatten()}, Target: {y.flatten()}, Output: {output} (Predicted Class: {network.np.argmax(output)})")


def test_gru_layer():
    from src.complex.layer import GRULayer

    print("\nTesting GRU layer integration...")
    layers: list[network.Layer] = [
        GRULayer(input_size=3, hidden_size=4),
        network.DenseLayer(input_size=4, output_size=2, activation='sigmoid')
    ]

    net = network.Network(layers)

    # input sequence: 5 steps, 3 features
    X_seq = network.np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2],
        [0.1, 0.0, 0.1],
        [0.2, 0.3, 0.1]
    ])

    y_seq = network.np.array([[1, 0]])  # dummy target for one-step classification

    out_seq = net.forward(X_seq)
    print("GRU output sequence shape:", out_seq.shape)
    print("GRU output last step:", out_seq[-1])

    # Make a dummy gradient matching output shape of Dense endpoint (1,2)
    grad = network.np.ones((1, 2)) * 0.1
    dx = net.backward(grad, eta=0.01)
    print("Backward pass produced dx shape:", dx.shape, "\n")


if __name__ == "__main__":
    #test_network()
    test_gru_layer()