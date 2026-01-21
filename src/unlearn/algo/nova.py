from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch import randn, Tensor
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
            forget_data: Tensor,
            forget_target: Tensor,
            optim: Optimizer,
            ):
        optim.zero_grad()

        # Make the noise batch
        f = [self.noise_samples]
        f.extend(forget_data.shape)
        noise_batch = randn(f) # Original vector of size 10
        
        forget_output = self.model(
            noise_batch.to(self.device),
        )

        forget_loss = - self.criterion(
            forget_output,
            forget_target.expand(self.noise_samples, -1), # extend the target too
        ) / self.noise_samples

        # Clip gradients to prevent explosion
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        forget_loss.backward()
        optim.step()

        return forget_loss.item()

    def unlearn(self, unlearn_data_loader: Any, test_loader: Optional[Any] = None) -> Module:
        """Runs the Gradient Difference unlearning process."""
        print(f"Starting Gradient Difference for {self.epochs} epoch(s).")
        optim = self._setup_optimizer()
        alpha = self.alpha
        num_epochs = self.epochs
        
        # Get ClearML Logger
        logger = None
        task = Task.current_task()
        if task:
            logger = task.get_logger()

        self.model.train()
        for epoch in range(num_epochs):
            total_retain_loss = 0.0
            total_forget_loss = 0.0
            
            for batch in unlearn_data_loader:
                # GD requires paired data: 'forget' and 'retain' (non-forget)
                forget_data = batch['forget']
                retain_data = batch['retain']
                
                # --- Forget Loss (L_F) ---
                f_data = forget_data['input'].to(self.device)
                f_target = forget_data['labels'].to(self.device)
                
                for forget_sample, forget_target in zip(f_data, f_target):
                    forget_loss = self.forget_loss(
                        forget_sample,
                        forget_target,
                        optim,
                    )
                    total_forget_loss += forget_loss

                # --- Retain Loss (L_R) ---
                r_data = retain_data['input'].to(self.device)
                r_target_indices = retain_data['labels'].to(self.device)

                optim.zero_grad()
                r_output = self.model(r_data)
                loss_r = self.criterion(r_output, r_target_indices)

                # --- Combined Loss (L_GD) ---
                # Total Loss to MINIMIZE: alpha * L_R -  L_F
                retain_loss = alpha * loss_r
                retain_loss.backward()

                # Clip gradients to prevent explosion
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()
                
                total_retain_loss += retain_loss.item()

            avg_forget_loss = total_forget_loss / len(unlearn_data_loader)
            avg_retain_loss = total_retain_loss / len(unlearn_data_loader)
            print(f"  GD Epoch {epoch+1}/{num_epochs}: Average Retain Loss: {avg_retain_loss:.4f}")
            print(f"  GD Epoch {epoch+1}/{num_epochs}: Average Forget Loss: {avg_forget_loss:.4f}")
            
            if logger:
                logger.report_scalar(title="Unlearning (GD)", series="Retain Loss", value=avg_retain_loss, iteration=epoch+1)
                logger.report_scalar(title="Unlearning (GD)", series="Forget Loss", value=avg_forget_loss, iteration=epoch+1)

            # Evaluate on Test Data if provided
            if test_loader:
                test_loss, test_acc = evaluate_loader(self.model, test_loader, self.device)
                self.model.train() # Switch back to train mode
                print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
                if logger:
                    logger.report_scalar(title="Unlearning (GD)", series="Test Loss", value=test_loss, iteration=epoch+1)
                    logger.report_scalar(title="Unlearning (GD)", series="Test Accuracy", value=test_acc, iteration=epoch+1)
            
        return self.model