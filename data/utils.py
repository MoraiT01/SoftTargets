import os
import yaml

from typing import Dict, Any

def is_ignored(pattern_to_check, gitignore_path='.gitignore'):
    """
    Checks if a pattern is present in the .gitignore file.
    
    Args:
        pattern_to_check (str): The string (e.g., 'logs/', '*.csv') to look for.
        gitignore_path (str): The path to the .gitignore file.
        
    Returns:
        bool: True if the pattern is found, False otherwise.
    """
    # Check if the .gitignore file exists
    if not os.path.exists(gitignore_path):
        print(f"⚠️ .gitignore file not found at {gitignore_path}")
        return False
        
    try:
        with open(gitignore_path, 'r') as f:
            # Read all lines and strip whitespace/newline characters from each line
            lines = [line.strip() for line in f if line.strip()]
            
            # Check for the pattern in the list of lines
            if pattern_to_check in lines:
                return True
            else:
                return False
                
    except IOError as e:
        print(f"❌ Error reading .gitignore file: {e}")
        return False
    
# Utility function to load architecture-specific config
def load_training_config(architecture: str) -> Dict[str, Any]:
    """Loads the training configuration based on the model architecture."""

    # Determine the file path based on the architecture
    arch = architecture.lower()
    config_filename = f"{arch}.yaml"
    config_path = os.path.join("configs", "training", config_filename)
    
    print(f"Attempting to load configuration for {architecture} from {config_path}...")
    
    try:
        if not os.path.exists(os.path.dirname(config_path)):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get("training", {})
            
    except FileNotFoundError:
        print(f"Warning: Configuration file not found at {config_path}. Using default hardcoded config.")
    except Exception as e:
        print(f"Warning: Failed to load config: {e}. Using default hardcoded config.")

    # Default/Fallback Configuration
    return {
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "save_path": f"saves/{arch}_trained_model.pth"
    }