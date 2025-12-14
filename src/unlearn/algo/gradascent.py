import torch
from torch.nn import Module

from typing import Any

from src.unlearn.base import BaseUnlearningAlgorithm

class GradientAscent(BaseUnlearningAlgorithm):
    """
    Implements the Gradient Ascent unlearning algorithm.
    This algorithm maximizes the loss on the forget set (L_F).
    """
    
    def unlearn(self, unlearn_data_loader: Any) -> Module:
        """Runs the Gradient Ascent unlearning process."""
        print(f"Starting Gradient Ascent for {self.config.get('epochs', 1)} epoch(s).")
        optimizer = self._setup_optimizer()
        
        self.model.train()
        num_epochs = self.config.get("epochs", 1)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in unlearn_data_loader:
                # Gradient Ascent uses only the 'forget' data from the paired batch
                forget_data = batch['forget']
                data = forget_data['input'].to(self.device)
                # Soft labels are already one-hot, convert to class indices for NLLLoss
                target_indices = forget_data['labels'].argmax(dim=1).to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                
                loss_f = - self.criterion(output, target_indices)
                
                # Gradient Ascent: Minimize -Loss (or maximize Loss). We use -loss_f.backward()
                loss_f.backward() 
                optimizer.step()
                total_loss += loss_f.item()
            
            print(f"  GA Epoch {epoch+1}/{num_epochs}: Average Loss: {total_loss / len(unlearn_data_loader):.4f}")
            
        return self.model