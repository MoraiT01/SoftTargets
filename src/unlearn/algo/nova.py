from torch.nn import Module
from torch.optim import Optimizer
from torch import randn
from typing import Any, Optional
from clearml import Task
from src.unlearn.base import BaseUnlearningAlgorithm
from src.eval.metrics import evaluate_loader

class NOVA(BaseUnlearningAlgorithm):
    """
    Docstring for NOVA
    """
    
    def forget_loss(
            self,
            forget_data,
            forget_target,
            optimizer: Optimizer,
            epoch: int,
            ):

        f = [8]
        f.extend(forget_data.squeeze(0).shape)
        noise_batch = randn(f) # Original vector of size 10

        forget_output = self.model(
            noise_batch.to(self.device),
        )

        forget_loss = - self.criterion(
            forget_output,
            forget_target,
        )

        optimizer.zero_grad()
        forget_loss.backward()
        optimizer.step()

        # Get ClearML Logger
        logger = None
        task = Task.current_task()
        if task:
            logger = task.get_logger()
        if logger:
            logger.report_scalar(title="Unlearning (GD)", series="Forget Loss", value=forget_loss, iteration=epoch+1)

    def unlearn(self, unlearn_data_loader: Any, test_loader: Optional[Any] = None) -> Module:
        """Runs the Gradient Difference unlearning process."""
        print(f"Starting Gradient Difference for {self.epochs} epoch(s).")
        optimizer = self._setup_optimizer()
        
        alpha = self.alpha
        self.model.train()
        num_epochs = self.epochs
        
        # Get ClearML Logger
        logger = None
        task = Task.current_task()
        if task:
            logger = task.get_logger()

        for epoch in range(num_epochs):
            total_combined_loss = 0.0
            
            for batch in unlearn_data_loader:
                # GD requires paired data: 'forget' and 'retain' (non-forget)
                forget_data = batch['forget']
                retain_data = batch['retain']
                
                # --- Forget Loss (L_F) ---
                f_data = forget_data['input'].to(self.device)
                f_target = forget_data['labels'].to(self.device)
                
                for forget_sample, forget_target in zip(f_data, f_target):
                    self.forget_loss(
                        forget_sample,
                        forget_target,
                        optimizer,
                        epoch=epoch,
                    )

                # --- Retain Loss (L_R) ---
                r_data = retain_data['input'].to(self.device)
                r_target_indices = retain_data['labels'].to(self.device)
                r_output = self.model(r_data)
                loss_r = self.criterion(r_output, r_target_indices)

                # --- Combined Loss (L_GD) ---
                # Total Loss to MINIMIZE: (1-alpha) * L_R - alpha * L_F
                combined_loss = alpha * loss_r

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                
                total_combined_loss += combined_loss.item()
            
            avg_loss = total_combined_loss / len(unlearn_data_loader)
            print(f"  GD Epoch {epoch+1}/{num_epochs}: Average Combined Loss: {avg_loss:.4f}")
            
            if logger:
                logger.report_scalar(title="Unlearning (GD)", series="Retain Loss", value=avg_loss, iteration=epoch+1)

            # Evaluate on Test Data if provided
            if test_loader:
                test_loss, test_acc = evaluate_loader(self.model, test_loader, self.device)
                self.model.train() # Switch back to train mode
                print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
                if logger:
                    logger.report_scalar(title="Unlearning (GD)", series="Test Loss", value=test_loss, iteration=epoch+1)
                    logger.report_scalar(title="Unlearning (GD)", series="Test Accuracy", value=test_acc, iteration=epoch+1)
            
        return self.model