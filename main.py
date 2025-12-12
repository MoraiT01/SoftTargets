import argparse
import src.training.train as train #
import src.unlearn.main as unlearn #
import src.eval.main as eval

# You can import from your 'src' folder because of the PYTHONPATH in the Dockerfile
from src.data.dataset_loaders import TrainTestDataset, UnlearningPairDataset
from data import create_csv, download_data

import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import Module
from clearml.automation.controller import PipelineDecorator

from src.training.models.cnn import CNN
from src.training.models.mlp import TwoLayerPerceptron
from src.eval.main import evaluate

from pathlib import Path
from typing import Dict, Any, Tuple
import yaml

# --- Define your Hyperparameters ---
HYPERPARAMS = {
    "dataset": "mnist",
    "mu_algo": "gradient_ascent",
    "architecture": "simple_cnn",
} # For now, this is just parked here; I don't know if I will need it in the end

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

# --- Define your pipeline components ---
@PipelineDecorator.component()
def data_loading() -> Path:
    try:
        path = download_data.main()
    except Exception as e:
        print(f"\nFATAL ERROR during data download: {e}")
        sys.exit(1)

    return path

@PipelineDecorator.component()
def data_preprocessing(args: Any, path: Path) -> Tuple[DataLoader, DataLoader, UnlearningPairDataset, DataLoader]:
    """
        return: Tuple of: Train Dataloader, Test Dataloader, UnlearningPairDataset, Retain Dataloader
    """
    # no need to create csv if it already exists
    if not os.path.exists(f"{path}/{args.dataset}_index.csv"):
        print(f"Creating CSV for {args.dataset}")
        create_csv.main([args.dataset])
    else:
        print(f"CSV for {args.dataset} already exists")

    # Load architecture-specific config to get the batch size
    temp_config = load_training_config(args.architecture)
    batch_size = temp_config.get('batch_size', 64)
    
    # Instantiate the Train Dataset
    train_ds    = TrainTestDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train") 
    
    # Create the training DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    test_ds     = TrainTestDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="test")
    test_dl     = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    unlearn_ds  = UnlearningPairDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train")

    retain_ds   = TrainTestDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train", sample_mode="retain")
    retain_dl   = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)

    return train_dl, test_dl, unlearn_ds, retain_dl

@PipelineDecorator.component()
def model_creation(architecute: str) -> Module:
    architecute = architecute.lower()
    print(f"Creating model with architecture: {architecute}")

    if architecute == "cnn":
        return CNN() #
    elif architecute == "mlp":
        return TwoLayerPerceptron() #
    else:
        raise ValueError(f"Unknown architecture: {architecute}. Options are 'cnn' and 'mlp'.")

@PipelineDecorator.component()
def training(train_loader: DataLoader, model: Module, architecture: str) -> Module:
    """
    Trains the provided model using the training data loader and architecture-specific configuration.
    """
    # Load architecture-specific training configuration
    train_config = load_training_config(architecture)
    
    if model is None:
        raise ValueError("Model object is missing.")
    
    print(f"Starting training for {architecture} with config: {train_config}")
    
    # Call the main training function from the 'train' module
    trained_model = train.train_model(
        model=model, 
        train_loader=train_loader, 
        config=train_config
    )
    
    return trained_model

@PipelineDecorator.component()
def evaluation(model: Module, args: Any, path: Path) -> Dict[str, float]: 
    """
    Evaluates the model on the test dataset. (Currently a placeholder)
    """
    
    # Load architecture-specific config to get the batch size
    temp_config = load_training_config(args.architecture)
    batch_size = temp_config.get('batch_size', 64)
    # NOTE: The loop over the range, similating the number of classes, will remain hard coded for now
    cls_dl = {
        i: DataLoader(
            dataset=TrainTestDataset(
                    csv_file=f"{path}/{args.dataset}_index.csv",
                    root_dir=".",
                    split="train",
                    classes=[str(i)]
                ),
            batch_size=batch_size,
            shuffle=False,
            ) for i in range(10)
        }
    
    result_dict = evaluate(model, cls_dl)

    return result_dict

@PipelineDecorator.component()
def unlearning(trained_model: Module, unlearn_ds: UnlearningPairDataset, mu_algo: str) -> Module:
    """
    Runs the specified machine unlearning algorithm on the trained model.
    
    Args:
        trained_model: The pre-trained model.
        unlearn_ds: The UnlearningPairDataset containing forget and retain data.
        mu_algo: The name of the unlearning algorithm (e.g., 'graddiff').
    """
    # Delegate the heavy lifting to the run_unlearning function in the unlearn module
    return unlearn.run_unlearning(trained_model, unlearn_ds, mu_algo)

def main():
    """
    Main function to parse arguments and run the training/unlearning pipeline.
    """
    parser = argparse.ArgumentParser(description="Run SoftTargets training and unlearning experiments.")
    
    # --- Define your command-line arguments ---
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="mnist",  # This is the default value
        help="The dataset to use (e.g., 'mnist', 'fashion_mnist')."
    )
    
    parser.add_argument(
        "--mu_algo", 
        type=str, 
        default="graddiff", # This is the default value
        help="The machine unlearning algorithm to use (e.g., 'graddiff', 'rmu')."
    )
    
    parser.add_argument(
        "--architecture", 
        type=str, 
        default="cnn", # This is the default value
        help="The model architecture to use (e.g., 'cnn', 'mlp')."
    )

    # --- Parse the arguments ---
    # Parse the arguments provided from the command line
    args = parser.parse_args()

    # --- Use the configurations in your script ---
    print("--- Starting Experiment ---")
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.mu_algo}")
    print(f"Architecture: {args.architecture}")
    print("---------------------------")

    # --- Run the pipeline ---
    ###
    # 1. Prepare the data and the model
    path = data_loading()
    train_dl, test_dl, unlearn_ds, retain_dl = data_preprocessing(args, path)
     
    # 2. Create the model
    model = model_creation(args.architecture)

    # 3a. Train the model
    trained_model = training(train_dl, model, args.architecture)

    # 3b. Train the baseline model
    # This one is only trained on the retain data, therefore it represents the ideal behaviour of the model after unlearning
    baseline_model = training(retain_dl, model, args.architecture)

    # 4. Evaluate the model
    evaluation_results = evaluation(trained_model, test_dl)
    print(f"Pre-Unlearning Evaluation Results: {evaluation_results}")

    # 5. Unlearn the model
    unlearned_model = unlearning(trained_model, unlearn_ds, args.mu_algo)

    # 6. Evaluate the unlearned model
    unlearned_evaluation_results = evaluation(unlearned_model, test_dl)
    print(f"Unlearned Evaluation Results: {unlearned_evaluation_results}")
    print("\nExperiment finished.")


if __name__ == "__main__":
    main()