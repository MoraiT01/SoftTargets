import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import Dict, Any, Optional
from clearml import Task
from src.eval.metrics import evaluate_loader

# --- Core Training Function ---

def train_epoch(model: Module, data_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Performs a single training epoch and returns the average loss."""
    model.train()
    running_loss = 0.0
    
    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(data_loader)
    # print(f"  | Epoch Loss: {avg_loss:.4f}") # Handled by main loop now
    return avg_loss

def train_model(model: Module, train_loader: DataLoader, config: Dict[str, Any], test_loader: Optional[DataLoader] = None) -> Module:
    """
    Main function to train the model and save the final checkpoint.
    
    Args:
        model: The model instance to be trained.
        train_loader: DataLoader for the training set.
        config: Dictionary of training hyperparameters.
        
    Returns:
        The trained model instance.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training model on device: {device}")
    
    # 2. Setup Optimizer and Loss Function
    lr = config.get("learning_rate", 0.001)
    optimizer_name = config.get("optimizer", "Adam").lower()
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        # Fallback to default if config is missing or invalid
        optimizer = optim.Adam(model.parameters(), lr=lr) 
        print(f"Warning: Unsupported optimizer '{optimizer_name}'. Falling back to Adam.")

    # Use NLLLoss because your models output log_softmax
    criterion = nn.CrossEntropyLoss()
    
    # Get ClearML Logger
    logger = None
    task = Task.current_task()
    if task:
        logger = task.get_logger()
    
    # 3. Training Loop
    num_epochs = config.get("epochs", 10)
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"--- Epoch {epoch}/{num_epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        print(f"  | Training Loss: {train_loss:.4f}")
        
        # Log Training Loss
        if logger:
            logger.report_scalar(title="Training", series="Loss", value=train_loss, iteration=epoch)

        # Evaluate on Test Data if provided
        if test_loader:
            test_loss, test_acc = evaluate_loader(model, test_loader, device)
            print(f"  | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
            if logger:
                logger.report_scalar(title="Training", series="Test Loss", value=test_loss, iteration=epoch)
                logger.report_scalar(title="Training", series="Test Accuracy", value=test_acc, iteration=epoch)
    
    return model