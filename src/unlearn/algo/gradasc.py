from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
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

        # I want to know the start Acc on the test set
        if test_loader:
            test_loss, test_acc = evaluate_loader(self.model, test_loader, self.device)
            print(f"### Starting Metrics ###")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
            print(f"### Starting Metrics ###")
            start_loss = test_loss
            start_acc  = test_acc
            if logger:
                logger.report_scalar(title="Unlearning (GA)", series="Test Loss", value=test_loss, iteration=0)
                logger.report_scalar(title="Unlearning (GA)", series="Test Accuracy", value=test_acc, iteration=0)
        ###

        self.model.train()
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

                # Clip gradients to prevent explosion
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss_f.item()
            
            avg_loss = total_loss / len(unlearn_data_loader)
            print(f"- Epoch {epoch+1}/{num_epochs}: Average Loss: {avg_loss:.4f}")
            
            if logger:
                logger.report_scalar(title="Unlearning (GA)", series="Forget Loss (Negative)", value=avg_loss, iteration=epoch+1)
            
            # Evaluate on Test Data if provided
            if test_loader:
                test_loss, test_acc = evaluate_loader(self.model, test_loader, self.device)
                self.model.train() # Switch back to train mode

                delta_loss = abs(test_loss - start_loss)
                delta_acc  = abs(test_acc  - start_acc )
                print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
                print(f"Change in Loss: {delta_loss:.4f} | Change in Acc: {delta_acc:.4f}")
                if logger:
                    logger.report_scalar(title="Unlearning (GA)", series="Test Loss", value=test_loss, iteration=epoch+1)
                    logger.report_scalar(title="Unlearning (GA)", series="Test Accuracy", value=test_acc, iteration=epoch+1)

                    logger.report_scaler(title="Metric Changes (GA)" , series="Test Loss", value=delta_loss, iteration=epoch+1)
                    logger.report_scaler(title="Metric Changes (GA)" , series="Test Accuracy", value=delta_acc, iteration=epoch+1)

        return self.model