import torch
from torch.nn import Module
from typing import Dict, Any, Union
import yaml
import os

# Import the custom classes and DataLoader from your project
from src.unlearn.algo import gradascent, graddiff
from src.data.dataset_loaders import UnlearningPairDataset, UnlearningDataLoader 
from clearml import Task

# Define a mapping from algorithm name to its class
ALGORITHM_MAP = {
    "gradasc": gradascent.GradientAscent,
    "graddiff": graddiff.GradientDifference,
}

def load_unlearning_config(algorithm: str) -> Dict[str, Any]:
    """Loads the unlearning configuration based on the algorithm name."""
    
    alg_name = algorithm.lower().replace(" ", "_")
    config_filename = f"{alg_name}.yaml"
    config_path = os.path.join("configs", "unlearning", config_filename)
    
    print(f"Attempting to load unlearning configuration for {algorithm} from {config_path}...")
    
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get("unlearning", {})
            
    except FileNotFoundError:
        print(f"Warning: Configuration file not found at {config_path}. Using default hardcoded config.")
    except Exception as e:
        print(f"Warning: Failed to load config: {e}. Using default hardcoded config.")

    # Default/Fallback Configuration
    if alg_name == "gradasc":
        return {"epochs": 1, "batch_size": 64, "learning_rate": 0.01, "optimizer": "SGD", "momentum": 0.9}
    elif alg_name == "graddiff":
        return {"epochs": 1, "batch_size": 64, "learning_rate": 0.001, "optimizer": "SGD", "momentum": 0.9, "alpha": 0.5}
    
    return {} # Return empty dict if algorithm is unknown

def run_unlearning(trained_model: Module, unlearn_ds: UnlearningPairDataset, algorithm_name: str) -> Module:
    """
    Selects, configures, and runs the specified unlearning algorithm.
    """
    alg_name_lower = algorithm_name.lower()
    
    if alg_name_lower not in ALGORITHM_MAP:
        raise ValueError(f"Unknown unlearning algorithm: {algorithm_name}. Options are {list(ALGORITHM_MAP.keys())}")

    # Load configuration
    config = load_unlearning_config(algorithm_name)
    
    # Instantiate DataLoader for unlearning (using the custom paired DataLoader)
    batch_size = config.get('batch_size', 64) 
    unlearn_dl = UnlearningDataLoader(unlearn_ds, batch_size=batch_size, shuffle=True)
    
    # Instantiate the algorithm class
    AlgorithmClass = ALGORITHM_MAP[alg_name_lower]
    unlearning_alg = AlgorithmClass(trained_model, config)
    
    print(f"Running unlearning using {AlgorithmClass.__name__}...")
    
    # Run the unlearning process
    unlearned_model = unlearning_alg.unlearn(unlearn_dl)
    
    # --- SAVE ONLY TO CLEARML ---
    filename = f"{alg_name_lower}_unlearned.pth"
    save_path = f"saves/{filename}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Save locally temporarily
    torch.save(unlearned_model.state_dict(), save_path)
    
    # 2. Upload to ClearML
    task = Task.current_task()
    if task:
        print(f"Uploading {filename} to ClearML...")
        task.upload_artifact(name="Unlearned Model", artifact_object=save_path)
        
        # 3. Remove local file to save space
        try:
            os.remove(save_path)
            print(f"Local file {save_path} removed.")
        except OSError as e:
            print(f"Error removing local file: {e}")
            
    unlearned_model.set_path(save_path) # Keep path for reference in code, even if file is gone
    
    return unlearned_model