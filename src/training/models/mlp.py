"""Simple Multi-Layer Perceptron (MLP) model for image classification."""

import torch

class TwoLayerPerceptron(torch.nn.Module):
    """
    A two layer perceptron model used for MNIST
    """
    def __init__(self):
        """
        Initializes the model
        """
        super(TwoLayerPerceptron, self).__init__()
        self.fc1 = torch.nn.Linear(784, 800)
        self.fc3 = torch.nn.Linear(800, 10)

        self.path = None

    def forward(self, x):
        """
        Defines the forward pass of the model
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x
    
    def set_path(self, new_path: str):
        """
        Sets the path to the model
        """
        self.path = new_path

    def get_path(self):
        """
        Gets the path to the model
        """
        return self.path
    
    __str__ = lambda self: "TwoLayerPerceptron"
