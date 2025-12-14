import torch
from torch.nn import Module
from typing import Dict, Any, Union
import yaml
import os
import time
from src.unlearn.algo import gradasc, graddiff
from src.data.dataset_loaders import UnlearningPairDataset, UnlearningDataLoader 
from clearml import Task

# Define a mapping from algorithm name to its class
ALGORITHM_MAP = {
    "gradasc": gradasc.GradientAscent,
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
        print(f"Warning: Failed to load config: {e}. Using default.")

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
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Load configuration
    config = load_unlearning_config(algorithm_name)
    batch_size = config.get('batch_size', 64) 
    unlearn_dl = UnlearningDataLoader(unlearn_ds, batch_size=batch_size, shuffle=True)
    
    # Instantiate the algorithm class
    AlgorithmClass = ALGORITHM_MAP[alg_name_lower]
    unlearning_alg = AlgorithmClass(trained_model, config)
    
    print(f"Running unlearning using {AlgorithmClass.__name__}...")
    
    # Run the unlearning process
    unlearned_model = unlearning_alg.unlearn(unlearn_dl)
    
    return unlearned_model