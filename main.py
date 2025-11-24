import argparse
import train  #
import unlearn #

# You can import from your 'src' folder because of the PYTHONPATH in the Dockerfile
from src.data.dataset_loaders import TrainTestDataset, UnlearningPairDataset
from data import create_csv, download_data

import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import Module
from clearml.automation.controller import PipelineDecorator

from typing import Dict

# --- Define your Hyperparameters ---
HYPERPARAMS = {
    "dataset": "mnist",
    "mu_algo": "gradient_ascent",
    "architecture": "simple_cnn",
} # For now, this is just parked here; I don't know if I will need it in the end

# --- Define your pipeline components ---
@PipelineDecorator.component()
def data_loading():
    try:
        download_data.main()
    except Exception as e:
        print(f"\nFATAL ERROR during data download: {e}")
        sys.exit(1)

@PipelineDecorator.component()
def data_preprocessing(dataset: str):
    """
        return: A Dictionary consiting of: Train Dataloader, Test Dataloader
    """
    # no need to create csv if it already exists
    if not os.path.exists(f"data/{dataset}_index.csv"):
        print(f"Creating CSV for {dataset}")
        create_csv.main([dataset])
    else:
        print(f"CSV for {dataset} already exists")

@PipelineDecorator.component()
def model_creation() -> Module:
    # TODO
    pass

@PipelineDecorator.component()
def training() -> Module:
    # TODO
    pass

@PipelineDecorator.component()
def evaluation() -> Dict[str, float]:
    # TODO
    pass

@PipelineDecorator.component()
def unlearning() -> Module:
    # TODO
    pass

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
        default="gradient_ascent", # This is the default value
        help="The machine unlearning algorithm to use (e.g., 'gradient_ascent', 'rmu')."
    )
    
    parser.add_argument(
        "--architecture", 
        type=str, 
        default="simple_cnn", # This is the default value
        help="The model architecture to use (e.g., 'simple_cnn', 'resnet18')."
    )

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
    # TODO
    # 1. Prepare the data and the model
    data_loading()
    data_preprocessing(args.dataset)
    train_dl    = TrainTestDataset(csv_file=f"data/{args.dataset}_index.csv", root_dir=f"data/{args.dataset}", split="train")
    test_dl     = TrainTestDataset(csv_file=f"data/{args.dataset}_index.csv", root_dir=f"data/{args.dataset}", split="test")
    model       = model_creation(args.architecture)
    # 2. Train the model
    # 3. Evaluate the model
    # 4. Unlearn the model
    # 5. Evaluate the unlearned model
    ###
    print("\nExperiment finished.")


if __name__ == "__main__":
    main()