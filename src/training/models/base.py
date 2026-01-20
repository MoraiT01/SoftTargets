import torch
from torch import nn

class StandardScaler(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
    
    def forward(self, x):
        # Add 1e-7 to prevent division by zero
        return (x - self.mean[None, :, None, None]) / (self.std[None, :, None, None] + 1e-7)

class BaseModel(nn.Module):
    def __init__(
            self,
            standard_scaler: StandardScaler = StandardScaler(torch.tensor([0., 0., 0.]), torch.tensor([1., 1., 1.]))
            ):
        super().__init__()

        self.standard_scaler = standard_scaler

    def scale(self, x):
        return self.standard_scaler(x)

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