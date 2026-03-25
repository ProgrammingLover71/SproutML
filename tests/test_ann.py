import numpy as np
from src.neural_network import NeuralNetwork, train


def test_ann_forward_shape():
    net = NeuralNetwork([2, 4, 1], activation_function='tanh', loss_function='mse')
    out = net.forward([0.5, -0.2])
    assert isinstance(out, np.ndarray)
    assert out.shape == (1,)


def test_ann_forward_values():
    net = NeuralNetwork([2, 2, 1], activation_function='sigmoid', loss_function='mse')
    out = net.forward([0.0, 0.0])
    assert 0.0 <= out[0] <= 1.0


def test_ann_training_reduces_loss():
    net = NeuralNetwork([2, 4, 1], activation_function='tanh', loss_function='mse')
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    # capture pre-training error
    pre = 0.0
    for xi, yi in zip(x, y):
        pred = net.forward(xi)
        pre += float(np.mean((pred - yi) ** 2))

    train(net, x.tolist(), y.tolist(), epochs=50, learning_rate=0.1, show_msg=False)

    post = 0.0
    for xi, yi in zip(x, y):
        pred = net.forward(xi)
        post += float(np.mean((pred - yi) ** 2))

    assert post < pre


if __name__ == '__main__':
    test_ann_forward_shape()
    test_ann_forward_values()
    test_ann_training_reduces_loss()
    print('ALL ANN TESTS PASSED')
