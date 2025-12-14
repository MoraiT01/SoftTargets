import torch
from typing import Dict, Any
from torch.nn import Module
from torch.utils.data import DataLoader

from typing import Optional

# Import the metric calculation functions
import src.eval.metrics as metrics
import src.eval.visualize as visualize

# ClearML Imports
from clearml import Task

def evaluate(model: Module, dataloaders: Dict[Any, DataLoader]) -> Dict[str, float]:
    """
    Dispatcher function that evaluates the model on a dictionary of DataLoaders.
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
        
    # Calculate overall averages
    if len(dataloaders) > 0:
        results["mean_loss"] = sum(results[f"class_{k}_loss"] for k in dataloaders) / len(dataloaders)
        results["mean_accuracy"] = sum(results[f"class_{k}_accuracy"] for k in dataloaders) / len(dataloaders)

    # --- CLEARML INTEGRATION ---
    # Get the current task (this will be the 'evaluation' pipeline step)
    task = Task.current_task()
    if task:
        logger = task.get_logger()
        # Log every metric in the results dictionary
        for metric_name, value in results.items():
            # We split the name to categorize it nicely in the UI (e.g., Accuracy/class_0)
            if "accuracy" in metric_name:
                series = metric_name.replace("_accuracy", "")
                logger.report_scalar(title="Accuracy", series=series, value=value, iteration=0)
            elif "loss" in metric_name:
                series = metric_name.replace("_loss", "")
                logger.report_scalar(title="Loss", series=series, value=value, iteration=0)
            else:
                logger.report_scalar(title="Metrics", series=metric_name, value=value, iteration=0)
    # ---------------------------

    return results

def compare_models(model_orig: Module, model_unlearned: Module) -> Dict[str, float]:
    """
    Compares two models to quantify the parameter changes.
    """
    print("Comparing model parameters...")
    
    # Delegate calculation to metrics.py
    avg_diff = metrics.calculate_parameter_difference(model_orig, model_unlearned)
    
    print(f"Average Parameter Change: {avg_diff:.6f}")
    
    # --- CLEARML INTEGRATION ---
    task = Task.current_task()
    if task:
        task.get_logger().report_scalar(
            title="Model_Diff", 
            series="parameter_change", 
            value=avg_diff, 
            iteration=0
        )
    # ---------------------------
    
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