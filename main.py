import argparse
import train  #
import unlearn #

# You can import from your 'src' folder because of the PYTHONPATH in the Dockerfile
from src.data.dataset_loaders import SoftTargetDataset

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

    # ** Your project logic goes here **
    # You can now pass these arguments to your train and unlearn functions.
    #
    # Example:
    # 1. Load data
    # print("Loading data...")
    # training_data = SoftTargetDataset(
    #     root_dir="data/softtarget_dataset",
    #     dataset_name=args.dataset,
    #     subset_to_include='full'
    # )
    
    # 2. Run training
    # print("Running training...")
    # model = train.run_training(training_data, architecture=args.architecture)
    
    # 3. Run unlearning
    # print("Running unlearning...")
    # unlearned_model = unlearn.run_unlearning(
    #     model, 
    #     algorithm=args.mu_algo
    # )
    
    print("\nExperiment finished.")


if __name__ == "__main__":
    main()