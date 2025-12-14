from torch.nn import Module

from typing import Any

from ..base import BaseUnlearningAlgorithm

class GradientDifference(BaseUnlearningAlgorithm):
    """
    Implements the Gradient Difference (GD) unlearning algorithm.
    Optimizes the combined loss: L_GD = (1 - alpha) * L_R - alpha * L_F
    """

    def unlearn(self, unlearn_data_loader: Any) -> Module:
        """Runs the Gradient Difference unlearning process."""
        print(f"Starting Gradient Difference for {self.config.get('epochs', 1)} epoch(s).")
        optimizer = self._setup_optimizer()
        
        alpha = self.config.get("alpha", 0.5)
        self.model.train()
        num_epochs = self.config.get("epochs", 1)

        for epoch in range(num_epochs):
            total_combined_loss = 0.0
            
            for batch in unlearn_data_loader:
                # GD requires paired data: 'forget' and 'retain' (non-forget)
                forget_data = batch['forget']
                retain_data = batch['retain']
                
                # --- Forget Loss (L_F) ---
                f_data = forget_data['input'].to(self.device)
                f_target_indices = forget_data['labels'].argmax(dim=1).to(self.device)
                f_output = self.model(f_data)
                loss_f = self.criterion(f_output, f_target_indices)

                # --- Retain Loss (L_R) ---
                r_data = retain_data['input'].to(self.device)
                r_target_indices = retain_data['labels'].argmax(dim=1).to(self.device)
                r_output = self.model(r_data)
                loss_r = self.criterion(r_output, r_target_indices)

                # --- Combined Loss (L_GD) ---
                # Total Loss to MINIMIZE: (1-alpha) * L_R - alpha * L_F
                combined_loss = (1 - alpha) * loss_r - alpha * loss_f

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                
                total_combined_loss += combined_loss.item()
            
            print(f"  GD Epoch {epoch+1}/{num_epochs}: Average Combined Loss: {total_combined_loss / len(unlearn_data_loader):.4f}")
            
        return self.model