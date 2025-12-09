import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import Dict, Any
import os

# --- Core Training Function ---

def train_epoch(model: Module, data_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Performs a single training epoch and returns the average loss."""
    model.train()
    running_loss = 0.0
    
    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        # Convert one-hot target tensor to class indices (required by nn.NLLLoss)
        target_indices = target.argmax(dim=1)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target_indices)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(data_loader)
    print(f"  | Epoch Loss: {avg_loss:.4f}")
    return avg_loss

def train_model(model: Module, train_loader: DataLoader, config: Dict[str, Any]) -> Module:
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
    criterion = nn.NLLLoss()
    
    # 3. Training Loop
    num_epochs = config.get("epochs", 10)
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"--- Epoch {epoch}/{num_epochs} ---")
        train_epoch(model, train_loader, optimizer, criterion, device)
        
    # 4. Save the model
    # save_path = config.get("save_path", "saves/trained_model.pth")
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # # Save the model state dictionary
    # torch.save(model.state_dict(), save_path)
    # # Set the path attribute on the model (defined in your CNN/MLP classes)
    # model.set_path(save_path)
    # print(f"\nModel saved to: {save_path}")
    
    return model