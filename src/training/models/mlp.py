"""Simple Multi-Layer Perceptron (MLP) model for image classification."""

import torch
from src.training.models.base import BaseModel, StandardScaler

class TwoLayerPerceptron(BaseModel):
    """
    A two layer perceptron model used for MNIST
    """
    def __init__(
            self,
            standard_scaler: StandardScaler = StandardScaler(torch.tensor([0., 0., 0.]), torch.tensor([1., 1., 1.])),
            ):
        """
        Initializes the model
        """
        super().__init__(standard_scaler=standard_scaler)
        
        self.fc1 = torch.nn.Linear(1024, 800)
        self.fc3 = torch.nn.Linear(800, 10)

        self.path = None

    def forward(self, x):
        """
        Defines the forward pass of the model
        """
        x = self.scale(x)
        # Flatten the tensor from [Batch, 1, H, W] to [Batch, H*W]
        x = x.view(x.size(0), -1) # Correctly flattens to [Batch, 784] for grayscale

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        
        return x
    
    __str__ = lambda self: "TwoLayerPerceptron"
