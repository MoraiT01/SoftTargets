import argparse
import src.training.train as train
from src.training.models.base import StandardScaler
import src.unlearn.main as unlearn

# You can import from your 'src' folder because of the PYTHONPATH in the Dockerfile
from src.data.dataset_loaders import TrainTestDataset, UnlearningPairDataset
from data import create_csv, download_data
from data.utils import load_training_config
from src.eval.visualize import plot_dataset_stats
from src.eval.main import evaluation

import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import Module
from clearml.automation.controller import PipelineDecorator

from src.training.models.cnn import CNN
from src.training.models.mlp import TwoLayerPerceptron
from src.eval.main import compare_models, visualize_pipeline_results, final_metrics_summary
from src.eval.aggregate_results import aggregate_runs

from typing import Dict, Any, Tuple, cast, Optional, Literal
import yaml

from clearml import Task
Task.set_offline(offline_mode=True)

# --- Define your Hyperparameters ---
HYPERPARAMS = {
    "dataset": "mnist",
    "mu_algo": "gradasc",
    "architecture": "mlp",
    "soft_targets": False,
}

# --- Define your pipeline components ---
@PipelineDecorator.component(cache=True, name="Download Data", return_values=["Data Path"])
def data_loading() -> str:
    try:
        path = download_data.main()
    except Exception as e:
        print(f"\nFATAL ERROR during data download: {e}")
        sys.exit(1)

    return str(path)

@PipelineDecorator.component(cache=True, name="Preprocess Data", return_values=["Train Dataloader", "Unlearning Dataset", "Retain Dataloader", "Test Dataloader"])
def data_preprocessing(args: Any, path: str, seed: int,) -> Tuple[DataLoader, UnlearningPairDataset, DataLoader, DataLoader]:
    """
        return: Tuple of: Train Dataloader, UnlearningPairDataset, Retain Dataloader, Test Dataloader
    """
    # no need to create csv if it already exists
    if not os.path.exists(f"{path}/{args.dataset}_index.csv"):
        print(f"Creating CSV for {args.dataset}")
        create_csv.main([args.dataset])
    else:
        print(f"CSV for {args.dataset} already exists")

    # Plot infos about the dataset
    try:
        df = pd.read_csv(f"{path}/{args.dataset}_index.csv")
        plot_dataset_stats(df, "f1_split")
    except Exception as e:
        print(f"Failed to plot dataset stats: {e}")

    # Create a generator for the DataLoader
    torch.manual_seed(seed=seed)
    g = torch.Generator()

    # Load architecture-specific config to get the batch size
    temp_config = load_training_config(args.architecture)
    batch_size = temp_config.get('batch_size', 64)
    
    # Instantiate the Train Dataset
    train_ds    = TrainTestDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train") 
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    
    unlearn_ds  = UnlearningPairDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train")

    retain_ds   = TrainTestDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="train", sample_mode="retain")
    retain_dl   = DataLoader(retain_ds, batch_size=batch_size, shuffle=True, generator=g)

    # Instantiate the Test Dataset
    test_ds = TrainTestDataset(csv_file=f"{path}/{args.dataset}_index.csv", root_dir=".", split="test")
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, generator=g)

    return train_dl, unlearn_ds, retain_dl, test_dl

@PipelineDecorator.component(cache=True, name="Create Startpoint", return_values=["Untrained Model"])
def model_creation(architecute: str, train_loader: DataLoader, seed: int) -> Module:
    architecute = architecute.lower()
    print(f"Creating model with architecture: {architecute}")

    data = torch.cat([b for b,_ in train_loader])
    mean = data.mean(dim=[0, 2, 3])
    std = data.std(dim=[0, 2, 3])
    mean_std = dict(mean=mean.cpu(), std=std.cpu())
    standard_scaler = StandardScaler(**mean_std)  

    torch.manual_seed(seed=seed)
    if architecute == "cnn":
        return CNN(standard_scaler=standard_scaler) 
    elif architecute == "mlp":
        return TwoLayerPerceptron(standard_scaler=standard_scaler) 
    else:
        raise ValueError(f"Unknown architecture: {architecute}. Options are 'cnn' and 'mlp'.")

@PipelineDecorator.component(cache=True, name = "Train Baseline", return_values=["Baseline Model", "Evaluation Results"])
def training_base(train_loader: DataLoader, model: Module, test_loader: DataLoader, args: Any) -> Tuple[Module, Dict[str, float]]:
    """
    Trains the provided model using the training data loader and architecture-specific configuration.
    """
    # Load architecture-specific training configuration
    train_config = load_training_config(args.architecture)
    
    if model is None:
        raise ValueError("Model object is missing.")
    
    print(f"Starting training for {args.architecture} with config: {train_config}")
    
    # Call the main training function from the 'train' module
    # Passing test_loader to enable progress monitoring
    baseline_model = train.train_model(
        model=model, 
        train_loader=train_loader, 
        config=train_config,
        test_loader=test_loader 
    )

    dataset = cast(TrainTestDataset, test_loader.dataset)

    evaluation_results = evaluation(baseline_model, architecture=args.architecture, path=dataset.csv_file)
    
    return baseline_model, evaluation_results

@PipelineDecorator.component(cache=True, name = "Train Target", return_values=["Trained Target Model", "Evaluation Results"])
def training_target(train_loader: DataLoader, model: Module, test_loader: DataLoader, args: Any) -> Tuple[Module, Dict[str, float]]:
    """
    Trains the provided model using the training data loader and architecture-specific configuration.
    """
    # Load architecture-specific training configuration
    train_config = load_training_config(args.architecture)
    
    if model is None:
        raise ValueError("Model object is missing.")

    print(f"Starting training for {args.architecture} with config: {train_config}")
    
    # Call the main training function from the 'train' module
    # Passing test_loader to enable progress monitoring
    target_model = train.train_model(
        model=model, 
        train_loader=train_loader, 
        config=train_config,
        test_loader=test_loader 
    )
    dataset = cast(TrainTestDataset, test_loader.dataset)
    evaluation_results = evaluation(target_model, architecture=args.architecture, path=dataset.csv_file)
    
    return target_model, evaluation_results

@PipelineDecorator.component(cache=False, name="Parameter Difference", return_values=["Evaluation Model Parameter Difference"])
def evaluation_difference(original_model: Module, unlearned_model: Module) -> Dict[str, float]:
    """
    Evaluates the quantitive parameter difference between the original and unlearned models.
    """
    result_dict = compare_models(original_model, unlearned_model)
    return result_dict

@PipelineDecorator.component(cache=False, name="Create Soft Targets", return_values=["Soft Targets Unlearn Dataset"])
def creating_soft_targets(model: Module, unlearn_ds: UnlearningPairDataset) -> UnlearningPairDataset:
    """
    Optional pipeline step to generate soft targets using the trained model.
    """
    print("Generating Soft Targets (Predicted Probabilities) for the Unlearning Dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    unlearn_ds.make_softtargets(model, device)
    return unlearn_ds

@PipelineDecorator.component(cache=False, name="Unlearning", return_values=["Unlearned Model", "Evaluation Results"])
def unlearning(
        target_model: Module,
        unlearn_ds: UnlearningPairDataset,
        test_loader: DataLoader,
        architecture: Literal["cnn", "mlp"],
        algorithm_name: Literal["gradasc", "graddiff", "nova"],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer: str,
        momentum: float,
        alpha: Optional[float] = None,
        noise_samples: Optional[int] = None,
        ) -> Tuple[Module, Dict[str, float]]:
    """
    Runs the specified machine unlearning algorithm on the trained model.
    """
    # Delegate the heavy lifting to the run_unlearning function in the unlearn module
    # Passing test_loader for monitoring
    unlearned_model =  unlearn.run_unlearning(
        target_model,
        unlearn_ds,
        algorithm_name=algorithm_name,
        test_loader=test_loader,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        momentum=momentum,
        alpha=alpha,
        noise_samples=noise_samples,
        )

    dataset = cast(TrainTestDataset, test_loader.dataset)
    evaluation_results = evaluation(unlearned_model, architecture=architecture, path=dataset.csv_file)
    
    return unlearned_model, evaluation_results

@PipelineDecorator.component(cache=False, name="Plotter")
def plotter(
    trained_res: Dict[str, float],
    base_res: Dict[str, float],
    unlearned_res: Dict[str, float],
    param_changes:Dict[str, float],
    args: Any,
    ):  
    """
    Calls the visualization module to generate and display/save visualizations.
    """
    # Reduce the results to 2 metrics
    # - Accuracy Distance to Baseline
    # - Total Change in Parameters
    acc_diff, param_change = final_metrics_summary(trained_res, base_res, unlearned_res, param_changes)
    # Visualization
    visualize_pipeline_results(trained_res, base_res, unlearned_res, param_changes)

    aggregate_runs(args, [acc_diff], [param_change])

@PipelineDecorator.pipeline(name="SoftTargets Pipeline", project="softtargets", version="3.2.4")
def main(args: Any):
    """
    Main function to parse arguments and run the training/unlearning pipeline.
    """
    
    # checking if the hyperparameters are supported
    if args.architecture not in ["cnn", "mlp"]:
        raise ValueError(f"Unsupported architecture: {args.architecture}. Options are 'cnn' and 'mlp'.")
    if args.mu_algo not in ["gradasc", "graddiff", "nova"]:
        raise ValueError(f"Unsupported machine unlearning algorithm: {args.mu_algo}. Options are 'gradasc', 'graddiff' and 'nova'.")
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
    train_dl, unlearn_ds, retain_dl, test_dl = data_preprocessing(args, path, seed=42)
     
    # 2. Create the model
    model = model_creation(args.architecture, train_dl, seed=42)

    # 3a. Train the model
    trained_model, trained_evaluation_results = training_base(train_dl, model, test_dl, args)

    # 3b. Train the baseline model (Retain-only)
    baseline_model, baseline_evaluation_results = training_target(retain_dl, model, test_dl, args)

    if args.softtargets:
        # 3c. Generate soft targets for the unlearning dataset
        unlearn_ds = creating_soft_targets(trained_model, unlearn_ds)

    # 4. Unlearn the model
    unlearn_config = yaml.load(open(f"configs/unlearn/{args.mu_algo}.yaml"), Loader=yaml.FullLoader)
    unlearned_model, unlearned_evaluation_results = unlearning(
        target_model=trained_model,
        unlearn_ds=unlearn_ds,
        test_loader=test_dl,
        architecture=args.architecture,
        algorithm_name=args.mu_algo,
        epochs=unlearn_config["unlearning"]["epochs"],
        batch_size=unlearn_config["unlearning"]["batch_size"],
        learning_rate=unlearn_config["unlearning"]["learning_rate"],
        optimizer=unlearn_config["unlearning"]["optimizer"],
        momentum=unlearn_config["unlearning"]["momentum"],
        alpha=unlearn_config["unlearning"]["alpha"] if "alpha" in unlearn_config["unlearning"] else None,
        noise_samples=unlearn_config["unlearning"]["noise_samples"] if "noise_samples" in unlearn_config["unlearning"] else None,
    )

    # 6. Evaluate differences between models
    difference = evaluation_difference(original_model=trained_model, unlearned_model=unlearned_model)
    print(f"Model Differences between trained and unlearned: {difference}")

    # 7. Ploting
    plotter(
        trained_res=trained_evaluation_results, 
        base_res=baseline_evaluation_results, 
        unlearned_res=unlearned_evaluation_results, 
        param_changes=difference,
        args=args,)
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