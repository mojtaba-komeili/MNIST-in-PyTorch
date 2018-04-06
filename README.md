# MNIST-in-PyTorch
PyTorch models for MNIST handwritten numbers dataset

This is a playground project for creating PyTorch modesl that solves MNIST digit recognition problem.
Plan is to create simple models for practice and learning purpose.
Models that are created so far are the following:

1- fc-MNIST: a fully connected neural net model with one hidden layer.

2- conv_net: a simple convnet from scratch. This neural network is similar to LeCun's original network for MNIST.

3- transfer_learn_resnet18: this model uses the trained model from ResNet. It operates in two modes:
* Retrain the whole model
* Fix the lower layers and only train the top layer
