# SproutML

SproutML (or Sprout) is a machine learning (ML) library that allows developers to easily implement artificial intelligence in their projects. It is designed with simplicity, flexibility, and performance in mind, making it suitable for both beginners and experienced practitioners.

The goal of SproutML is to reduce the complexity typically associated with machine learning by providing intuitive APIs, clear abstractions, and minimal setup. Whether you're building a simple predictive model or experimenting with more advanced techniques, Sprout aims to streamline the development process.

# Key Features

- Easy-to-use and consistent API
- Modular architecture for extensibility
- Support for common ML algorithms out of the box
- Lightweight and fast, with minimal dependencies
- Designed for rapid prototyping and production use

SproutML focuses on developer experience, allowing you to spend less time dealing with boilerplate and more time building intelligent systems.

# Installation

Sprout can be installed in just one step:
```bash
py -m pip install sproutml
```

# Quick Example

The following snippet shows how to create, use and train a small feed-forward neural network:

```py
from sproutml import NeuralNetwork, train

net = NeuralNetwork(
    [2, 3, 1],                      # The network will have 2 inputs, 3 hidden nodes and one output
    activation_function = 'tanh',   # Sigmoid, ReLU, SiLU and GeLU are also supported
    loss_function = 'cross-entropy' # MSE supported too
)
```

# Philosophy

SproutML is built around a few core principles:

- Simplicity over complexity
- Readability over cleverness
- Practicality over completeness

# Roadmap

- Add support for more algorithms
- Improve performance optimizations
- Expand documentation and examples
- Introduce model serialization and deployment tools

# Contributing

Contributions are welcome. If you'd like to help improve SproutML, feel free to open issues or submit pull requests.

# License

Copyright (c) 2026 steak

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.