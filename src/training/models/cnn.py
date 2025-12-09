"""Simple Convolutional Neural Network (CNN) model for image classification."""

import torch

class CNN(torch.nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input channels=3 for RGB
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 128)  # Adjust input dim for 28x28 image after 3 layers of pooling
        self.fc2 = torch.nn.Linear(128, 10)

        self.path = None
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU and softmax
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        
        return x
    
    def set_path(self, new_path: str):
        self.path = new_path

    def get_path(self):
        return self.path

    __str__ = lambda self: "CNN"