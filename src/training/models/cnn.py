"""Simple Convolutional Neural Network (CNN) model for image classification."""

import torch
from src.training.models.base import BaseModel, StandardScaler

class CNN(BaseModel):
    def __init__(
            self,
            standard_scaler: StandardScaler = StandardScaler(torch.tensor([0., 0., 0.]), torch.tensor([1., 1., 1.])),
            ):
        super(CNN, self).__init__(standard_scaler=standard_scaler)
        
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input channels=3 for RGB
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 128)  # Adjust input dim for 28x28 image after 3 layers of pooling
        self.fc2 = torch.nn.Linear(128, 10)

        self.path = None
    
    def forward(self, x):

        x = self.scale(x)
        # Convolutional layers with ReLU and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU and softmax
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    __str__ = lambda self: "CNN"