from torch.nn import Module
from typing import Any, Optional
from clearml import Task
from src.unlearn.base import BaseUnlearningAlgorithm
from src.eval.metrics import evaluate_loader

class GradientAscent(BaseUnlearningAlgorithm):
    """
    Implements the Gradient Ascent unlearning algorithm.
    This algorithm maximizes the loss on the forget set (L_F).
    """
    
    def unlearn(self, unlearn_data_loader: Any, test_loader: Optional[Any] = None) -> Module:
        """Runs the Gradient Ascent unlearning process."""
        print(f"Starting Gradient Ascent for {self.epochs} epoch(s).")
        optimizer = self._setup_optimizer()
        
        self.model.train()
        num_epochs = self.epochs
        
        # Get ClearML Logger
        logger = None
        task = Task.current_task()
        if task:
            logger = task.get_logger()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in unlearn_data_loader:
                # Gradient Ascent uses only the 'forget' data from the paired batch
                forget_data = batch['forget']
                data = forget_data['input'].to(self.device)
                # Soft labels are already one-hot, convert to class indices for NLLLoss
                target_indices = forget_data['labels'].to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                
                loss_f = - self.criterion(output, target_indices)
                
                # Gradient Ascent: Minimize -Loss (or maximize Loss). We use -loss_f.backward()
                loss_f.backward() 
                optimizer.step()
                total_loss += loss_f.item()
            
            avg_loss = total_loss / len(unlearn_data_loader)
            print(f"  GA Epoch {epoch+1}/{num_epochs}: Average Loss: {avg_loss:.4f}")
            
            if logger:
                logger.report_scalar(title="Unlearning (GA)", series="Forget Loss (Negative)", value=avg_loss, iteration=epoch+1)
            
            # Evaluate on Test Data if provided
            if test_loader:
                test_loss, test_acc = evaluate_loader(self.model, test_loader, self.device)
                self.model.train() # Switch back to train mode
                print(f"       | Test Acc: {test_acc:.4f}")
                if logger:
                    logger.report_scalar(title="Unlearning (GA)", series="Test Accuracy", value=test_acc, iteration=epoch+1)

        return self.model