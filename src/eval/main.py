import torch
from typing import Dict, Any
from torch.nn import Module
from torch.utils.data import DataLoader

from typing import Optional

# Import the metric calculation functions
import src.eval.metrics as metrics
import src.eval.visualize as visualize

def evaluate(model: Module, dataloaders: Dict[Any, DataLoader]) -> Dict[str, float]:
    """
    Dispatcher function that evaluates the model on a dictionary of DataLoaders.
    
    Args:
        model: The PyTorch model to evaluate.
        dataloaders: A dictionary where keys are identifiers (e.g., class names or indices) 
                     and values are DataLoaders.
                     
    Returns:
        Dict[str, float]: A dictionary containing loss and accuracy for each loader.
                          Keys will be formatted as "{key}_loss" and "{key}_accuracy".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = {}
    
    print(f"Starting evaluation on {len(dataloaders)} split(s)...")
    
    for key, loader in dataloaders.items():
        # Delegate the actual calculation to metrics.py
        avg_loss, accuracy = metrics.evaluate_loader(model, loader, device)
        
        # Store results with descriptive keys
        results[f"class_{key}_loss"] = avg_loss
        results[f"class_{key}_accuracy"] = accuracy
        
        # Optional: Print per-split progress (helpful for debugging)
        # print(f"  Split '{key}': Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

    # Calculate overall averages across all splits (optional, but often useful)
    if len(dataloaders) > 0:
        results["mean_loss"] = sum(results[f"class_{k}_loss"] for k in dataloaders) / len(dataloaders)
        results["mean_accuracy"] = sum(results[f"class_{k}_accuracy"] for k in dataloaders) / len(dataloaders)

    return results

def compare_models(model_orig: Module, model_unlearned: Module) -> Dict[str, float]:
    """
    Compares two models to quantify the parameter changes.
    
    Args:
        model_orig: The original trained model.
        model_unlearned: The model after the unlearning process.
        
    Returns:
        Dict[str, float]: A dictionary containing the average parameter change.
    """
    print("Comparing model parameters...")
    
    # Delegate calculation to metrics.py
    avg_diff = metrics.calculate_parameter_difference(model_orig, model_unlearned)
    
    print(f"Average Parameter Change: {avg_diff:.6f}")
    
    return {
        "avg_parameter_change": avg_diff
    }

def visualize_pipeline_results(
    trained_res: Dict[str, float],
    base_res: Dict[str, float],
    unlearned_res: Dict[str, float],
    param_changes: Optional[Dict[str, float]] = None
):
    """
    Wrapper to call the visualization module from the main pipeline.
    """
    print("\n--- Generating Visualizations ---")
    visualize.visualize_all(trained_res, base_res, unlearned_res, param_changes)