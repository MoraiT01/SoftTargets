import re
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Literal

def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> int:
    """
    Calculates the number of correct predictions.
    
    Args:
        output: Model output (logits or log_softmax), shape [Batch, NumClasses]
        target: Target class indices, shape [Batch]
        
    Returns:
        int: Number of correct predictions
    """
    # Get the index of the max log-probability
    preds = output.argmax(dim=1, keepdim=True)
    # Compare with target (view_as ensures shapes match)
    correct = preds.eq(target.view_as(preds)).sum().item()
    return int(correct)

def evaluate_loader(model: Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on a single DataLoader.
    
    Args:
        model: The model to evaluate.
        data_loader: The DataLoader containing the data.
        device: The device to run evaluation on.
        
    Returns:
        Tuple[float, float]: (Average Loss, Average Accuracy)
    """
    model.eval()
    # Using NLLLoss consistent with the training (assuming model outputs log_softmax)
    criterion = nn.NLLLoss() 
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # The TrainTestDataset returns one-hot encoded targets.
            # nn.NLLLoss expects class indices (LongTensor).
            if target.dim() > 1 and target.size(1) > 1:
                target_indices = target.argmax(dim=1)
            else:
                target_indices = target
            
            output = model(data)
            
            # Calculate Loss (summed up, to average correctly over total samples later)
            loss = criterion(output, target_indices)
            total_loss += loss.item() * data.size(0)
            
            # Calculate Accuracy
            total_correct += calculate_accuracy(output, target_indices)
            total_samples += data.size(0)
            
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy

def calculate_parameter_difference(model_a: Module, model_b: Module) -> float:
    """
    Calculates the average absolute difference between the parameters of two models.
    
    Args:
        model_a: The first model (e.g., original trained model).
        model_b: The second model (e.g., unlearned model).
        
    Returns:
        float: The average absolute difference per parameter element.
    """
    total_diff = 0.0
    total_params = 0
    
    # Ensure models are on the same device (CPU is usually sufficient for this comparison)
    device = torch.device("cpu")
    model_a.to(device)
    model_b.to(device)
    
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    
    with torch.no_grad():
        for name, param_a in params_a.items():
            if name in params_b:
                param_b = params_b[name]
                
                # Check for shape mismatch just in case
                if param_a.shape != param_b.shape:
                    print(f"Warning: Shape mismatch for parameter '{name}'. Skipping.")
                    continue
                
                # Calculate absolute difference sum for this layer
                diff = (param_a - param_b).abs().sum().item()
                
                total_diff += diff
                total_params += param_a.numel()
            else:
                print(f"Warning: Parameter '{name}' found in model_a but not in model_b.")

    if total_params == 0:
        return 0.0
        
    return total_diff / total_params

def accuracy_distance(results_a: Dict[str, float], results_b: Dict[str, float], metric: Literal["accuracy", "loss"]="accuracy") -> float:
    """
    Calculates the average absolute difference in accuracy between two result dictionaries.
    
    Args:
        results_a: Dictionary containing accuracy metrics (e.g., from model A).
        results_b: Dictionary containing accuracy metrics (e.g., from model B).
    """
    total_diff = 0.0
    count = 0
    
    # Pattern matches keys like "class_0_accuracy" or "class_cat_loss"
    # Captures the middle part as the label
    pattern = re.compile(fr"class_(.+)_{metric}")
    
    for key, value in results_a.items():
        match = pattern.match(key)
        if match:
            # label = match.group(1)
            # Extracted Difference for this label
            if key in results_b:
                diff = abs(value - results_b[key])
                total_diff += diff
                count += 1
            else:
                print(f"Warning: Key '{key}' found in results_a but not in results_b.")

    return total_diff / count if count > 0 else 0.0