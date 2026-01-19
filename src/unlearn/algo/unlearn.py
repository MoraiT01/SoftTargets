from torch.nn import Module
from typing import Any, Optional
from clearml import Task
from src.unlearn.base import BaseUnlearningAlgorithm
from src.eval.metrics import evaluate_loader

class NOVA(BaseUnlearningAlgorithm):
    """
    Docstring for NOVA
    """
    
    def unlearn(self, unlearn_data_loader: Any, test_loader: Optional[Any] = None) -> Module:
        """Runs the NOVA unlearning process."""
        
        pass