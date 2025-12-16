import argparse
import os
import json
from typing import Any, List
try:
    from src.eval.visualize import plot_distributions
except ImportError:
    from visualize import plot_distributions

def aggregate_runs(
        args: Any,
        accuracies: List[float] = None, 
        param_changes: List[float] = None,
        json_file_path: str = "experiment_results.json", 
        target_runs: int = 30):
    
    # 2. JSON Logging Logic
    json_file_path = "experiment_results.json"
    
    # Initialize data structure
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # Define keys for nested structure
    arch = args.architecture
    algo = args.mu_algo
    dset = args.dataset
    st_key = f"softtargets_{args.softtargets}"
    
    # Ensure structure exists
    if arch not in data: data[arch] = {}
    if algo not in data[arch]: data[arch][algo] = {}
    if dset not in data[arch][algo]: data[arch][algo][dset] = {}
    if st_key not in data[arch][algo][dset]: 
        data[arch][algo][dset][st_key] = {
            "accuracies": [], 
            "param_changes": []
        }

    target_entry = data[arch][algo][dset][st_key]
    if accuracies is not None and param_changes is not None:
        # Append new results
        
        target_entry["accuracies"].extend(accuracies)
        target_entry["param_changes"].extend(param_changes)
        
    # Save back to file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Logged results to {json_file_path}. Total runs for this config: {len(target_entry['accuracies'])}")

    # 3. Check for 30 runs to trigger Distribution Analysis
    if len(target_entry["accuracies"]) == target_runs:
        print(">>> 30 runs reached! Executing Distribution Analysis Task... <<<")
        
        context_title = f"{arch}_{algo}_{dset}_{st_key}"
        plot_distributions(
            accuracies=target_entry["accuracies"], 
            param_changes=target_entry["param_changes"],
            context_title=context_title
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--mu_algo", type=str, default="gradasc")
    parser.add_argument("--architecture", type=str, default="mlp")
    parser.add_argument("--softtargets", action="store_true")
    
    args = parser.parse_args()
    
    aggregate_runs(args)