import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from typing import Dict, Any, Optional
import copy

class BaseUnlearningAlgorithm:
    """Base class for all machine unlearning algorithms."""
    
    def __init__(
            self,
            model: Module,
            epochs: int = 5,
            batch_size: int = 32,
            learning_rate: float = 0.00005,
            optimizer: str = "SGD",
            momentum: float = 0.9,
            alpha: Optional[float] = None,
            noise_samples: Optional[int] = None,
            ):
        """
        Initializes the algorithm with the model and configuration.
        A deep-copy of the model is made to ensure the original trained model is preserved.
        """
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.alpha = alpha
        self.noise_samples = noise_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = copy.deepcopy(model).to(self.device)
        # Assuming model output is log_softmax, so we use NLLLoss
        self.criterion = nn.CrossEntropyLoss()
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Sets up the optimizer based on the config."""
        lr = self.learning_rate
        optimizer_name = self.optimizer
        
        # Get optional parameters
        momentum = self.momentum
        weight_decay = 1e-4
        
        if optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            print(f"Warning: Unsupported unlearning optimizer '{optimizer_name}'. Falling back to SGD.")
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            
    def unlearn(self, unlearn_data_loader: Any, test_loader: Optional[Any] = None) -> Module:
        """
        The main unlearning method to be implemented by subclasses.
        Args:
            unlearn_data_loader: A DataLoader tailored for the algorithm.
            test_loader: Optional DataLoader for evaluating progress during unlearning.
        
        Returns:
            The unlearned model.
        """
        raise NotImplementedError("Subclasses must implement the 'unlearn' method.")