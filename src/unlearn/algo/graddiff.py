from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from typing import Any, Optional
from clearml import Task
from src.unlearn.base import BaseUnlearningAlgorithm
from src.eval.metrics import evaluate_loader

class GradientDifference(BaseUnlearningAlgorithm):
    """
    Implements the Gradient Difference (GD) unlearning algorithm.
    Optimizes the combined loss: L_GD = (1 - alpha) * L_R - alpha * L_F
    """

    def unlearn(self, unlearn_data_loader: Any, test_loader: Optional[Any] = None) -> Module:
        """Runs the Gradient Difference unlearning process."""
        print(f"Starting Gradient Difference for {self.epochs} epoch(s).")
        optimizer = self._setup_optimizer()
        
        alpha = self.alpha
        num_epochs = self.epochs
        
        # Get ClearML Logger
        logger = None
        task = Task.current_task()
        if task:
            logger = task.get_logger()

        # I want to know the start Acc on the test set
        if test_loader:
            test_loss, test_acc = evaluate_loader(self.model, test_loader, self.device)
            print(f"### Starting Metrics ###")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
            print(f"### Starting Metrics ###")
            start_loss = test_loss
            start_acc  = test_acc
            if logger:
                logger.report_scalar(title="Unlearning (GD)", series="Test Loss", value=test_loss, iteration=0)
                logger.report_scalar(title="Unlearning (GD)", series="Test Accuracy", value=test_acc, iteration=0)
        ###

        self.model.train()
        for epoch in range(num_epochs):
            total_combined_loss = 0.0
            
            for batch in unlearn_data_loader:
                # GD requires paired data: 'forget' and 'retain' (non-forget)
                forget_data = batch['forget']
                retain_data = batch['retain']
                
                # --- Forget Loss (L_F) ---
                f_data = forget_data['input'].to(self.device)
                f_target_indices = forget_data['labels'].to(self.device)
                f_output = self.model(f_data)
                loss_f = self.criterion(f_output, f_target_indices)

                # --- Retain Loss (L_R) ---
                r_data = retain_data['input'].to(self.device)
                r_target_indices = retain_data['labels'].to(self.device)
                r_output = self.model(r_data)
                loss_r = self.criterion(r_output, r_target_indices)

                # --- Combined Loss (L_GD) ---
                # Total Loss to MINIMIZE: (1-alpha) * L_R - alpha * L_F
                combined_loss = alpha * loss_r - loss_f

                optimizer.zero_grad()
                combined_loss.backward()

                # Clip gradients to prevent explosion
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_combined_loss += combined_loss.item()
            
            avg_loss = total_combined_loss / len(unlearn_data_loader)
            print(f"- Epoch {epoch+1}/{num_epochs}: Average Combined Loss: {avg_loss:.4f}")
            
            if logger:
                logger.report_scalar(title="Unlearning (GD)", series="Combined Loss", value=avg_loss, iteration=epoch+1)

            # Evaluate on Test Data if provided
            if test_loader:
                test_loss, test_acc = evaluate_loader(self.model, test_loader, self.device)
                self.model.train() # Switch back to train mode

                delta_loss = abs(test_loss - start_loss)
                delta_acc  = abs(test_acc  - start_acc )
                print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
                print(f"Change in Loss: {delta_loss:.4f} | Change in Acc: {delta_acc:.4f}")
                if logger:
                    logger.report_scalar(title="Unlearning (GD)", series="Test Loss", value=test_loss, iteration=epoch+1)
                    logger.report_scalar(title="Unlearning (GD)", series="Test Accuracy", value=test_acc, iteration=epoch+1)

                    logger.report_scalar(title="Metric Changes (GD)" , series="Test Loss", value=delta_loss, iteration=epoch+1)
                    logger.report_scalar(title="Metric Changes (GD)" , series="Test Accuracy", value=delta_acc, iteration=epoch+1)
            
        return self.model