import argparse
import src.training.train as train
import src.unlearn.main as unlearn

# You can import from your 'src' folder because of the PYTHONPATH in the Dockerfile
from src.data.dataset_loaders import TrainTestDataset, UnlearningPairDataset
from data import create_csv, download_data
from data.utils import load_training_config
from src.eval.visualize import plot_dataset_stats

import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import Module
from clearml.automation.controller import PipelineDecorator

from src.training.models.cnn import CNN
from src.training.models.mlp import TwoLayerPerceptron
from src.eval.main import evaluate, compare_models, visualize_pipeline_results

from typing import Dict, Any, Tuple

# --- Define your Hyperparameters ---
HYPERPARAMS = {
    "dataset": "mnist",
    "mu_algo": "gradasc",
    "architecture": "mlp",
    "soft_targets": False,
} # For now, this is just parked here; I don't know if I will need it in the end

# --- Define your pipeline components ---
@PipelineDecorator.component(cache=True, return_values=["Data Path"])
def data_loading() -> str:
    try:
        path = download_data.main()
    except Exception as e:
        print(f"\nFATAL ERROR during data download: {e}")
        sys.exit(1)

    return str(path)

@PipelineDecorator.component(cache=True, return_values=["Train Dataloader", "Unlearning Dataset", "Retain Dataloader"])
def data_preprocessing(args: Any, path: str) -> Tuple[DataLoader, UnlearningPairDataset, DataLoader]:
    """
        return: Tuple of: Train Dataloader, Test Dataloader, UnlearningPairDataset, Retain Dataloader
    """
    # no need to create csv if it already exists
    if not os.path.exists(f"{path}/{args.dataset}_index.csv"):
        print(f"Creating CSV for {args.dataset}")
        create_csv.main([args.dataset])
    else:
        print(f"CSV for {args.dataset} already exists")

    # Plot infos about the dataset
    # Firstly get the dataframe
    try:
        df = pd.read_csv(f"{path}/{args.dataset}_index.csv")
        plot_dataset_stats(df, "f1_split")
    except Exception as e:
        print(f"Failed to plot dataset stats: {e}")

    # Load architecture-specific config to get the batch size
    temp_config = load_training_config(args.architecture)
    batch_size = temp_config.get('batch_size', 64)
    
    # Instantiate the Train Dataset
    train_ds    = TrainTestDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train") 
    
    # Create the training DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    unlearn_ds  = UnlearningPairDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train")

    retain_ds   = TrainTestDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train", sample_mode="retain")
    retain_dl   = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)

    return train_dl, unlearn_ds, retain_dl

@PipelineDecorator.component(cache=True, return_values=["Untrained Model"])
def model_creation(architecute: str) -> Module:
    architecute = architecute.lower()
    print(f"Creating model with architecture: {architecute}")

    if architecute == "cnn":
        return CNN() #
    elif architecute == "mlp":
        return TwoLayerPerceptron() #
    else:
        raise ValueError(f"Unknown architecture: {architecute}. Options are 'cnn' and 'mlp'.")

@PipelineDecorator.component(cache=False, return_values=["Trained Model"])
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

@PipelineDecorator.component(cache=False, return_values=["Evaluated Model"])
def evaluation(model: Module, args: Any, path: str) -> Dict[str, float]: 
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

@PipelineDecorator.component(cache=False, return_values=["Evaluation Model Parameter Difference"])
def evaluation_difference(original_model: Module, unlearned_model: Module) -> Dict[str, float]:
    """
    Evaluates the quantitive parameter difference between the original and unlearned models.
    """

    result_dict = compare_models(original_model, unlearned_model)

    return result_dict

@PipelineDecorator.component(cache=False, return_values=["Soft Targets Unlearn Dataset"])
def creating_soft_targets(model: Module, unlearn_ds: UnlearningPairDataset) -> UnlearningPairDataset:
    """
    Optional pipeline step to generate soft targets using the trained model.
    """
    
    print("Generating Soft Targets (Predicted Probabilities) for the Unlearning Dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure model is on the correct device
    model.to(device)
    
    # Call the dataset method to populate soft targets
    unlearn_ds.make_softtargets(model, device)
        
    return unlearn_ds

@PipelineDecorator.component(cache=False, return_values=["Unlearned Model"])
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

@PipelineDecorator.component(cache=False)
def plotter(
    trained_res: Dict[str, float],
    base_res: Dict[str, float],
    unlearned_res: Dict[str, float],
    param_changes:Dict[str, float],
    ):  
    """
        Calls the visualization module to generate and display/save visualizations.
    """
    visualize_pipeline_results(trained_res, base_res, unlearned_res, param_changes)

@PipelineDecorator.pipeline(name="SoftTargets Pipeline", project="softtargets", version="1.3.2")
def main(args: Any):
    """
    Main function to parse arguments and run the training/unlearning pipeline.
    """
    
    # checking if the hyperparameters are supported
    if args.architecture not in ["cnn", "mlp"]:
        raise ValueError(f"Unsupported architecture: {args.architecture}. Options are 'cnn' and 'mlp'.")
    if args.mu_algo not in ["gradasc", "graddiff"]:
        raise ValueError(f"Unsupported machine unlearning algorithm: {args.mu_algo}. Options are 'gradasc' and 'graddiff'.")
    if args.dataset not in ["mnist", "fashion_mnist"]:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Options are 'mnist' and 'fashion_mnist'.")
    
    # --- Use the configurations in your script ---
    print("--- Starting Experiment ---")
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.mu_algo}")
    print(f"Architecture: {args.architecture}")
    print(f"Soft Targets: {args.softtargets}")
    print("---------------------------")

    # --- Run the pipeline ---
    ###
    # 1. Prepare the data and the model
    path = data_loading()
    train_dl, unlearn_ds, retain_dl = data_preprocessing(args, path)
     
    # 2. Create the model
    model = model_creation(args.architecture)

    # 3a. Train the model
    trained_model = training(train_dl, model, args.architecture)

    # 3b. Train the baseline model
    # This one is only trained on the retain data, therefore it represents the ideal behaviour of the model after unlearning
    baseline_model = training(retain_dl, model, args.architecture)

    if args.softtargets:
        # 3c. Generate soft targets for the unlearning dataset
        unlearn_ds = creating_soft_targets(trained_model, unlearn_ds)

    # 4. Unlearn the model
    unlearned_model = unlearning(trained_model, unlearn_ds, args.mu_algo)

    # 5a. Evaluate Baseline Model
    baseline_evaluation_results = evaluation(baseline_model, args=args, path=path)
    print(f"Baseline Evaluation Results: {baseline_evaluation_results}")

    # 5b. Evaluate the trained model
    trained_evaluation_results = evaluation(trained_model, args=args, path=path)
    print(f"Trained Evaluation Results: {trained_evaluation_results}")

    # 5c. Evaluate the unlearned model
    unlearned_evaluation_results = evaluation(unlearned_model, args=args, path=path)
    print(f"Unlearned Evaluation Results: {unlearned_evaluation_results}")

    # 6. Evaluate differences between models
    difference = evaluation_difference(original_model=trained_model, unlearned_model=unlearned_model)
    print(f"Model Differences between trained and unlearned: {difference}")

    # 7. Ploting
    plotter(trained_res=trained_evaluation_results, base_res=baseline_evaluation_results, unlearned_res=unlearned_evaluation_results, param_changes=difference)

    print("\nExperiment finished.")


if __name__ == "__main__":
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

    parser.add_argument(
        "--softtargets",
        action="store_true", # If flag is present, value is True. Default is False.
        help="Whether to use soft targets (predicted probabilities) instead of hard labels for unlearning."
    )

    # --- Parse the arguments ---
    # Parse the arguments provided from the command line
    args = parser.parse_args()

    # PipelineDecorator.set_default_execution_queue("default")
    PipelineDecorator.run_locally()
    main(args)