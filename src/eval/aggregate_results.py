import argparse
import numpy as np
import pandas as pd
from clearml import Task
import plotly.graph_objects as go

def aggregate_runs(project_name, dataset, mu_algo, architecture, softtargets):
    print(f"Searching for tasks in project '{project_name}'...")
    print(f"Filters: Dataset={dataset}, Algo={mu_algo}, Arch={architecture}, SoftTargets={softtargets}")

    # 1. Search for Evaluation Tasks
    # The 'evaluation' function in the pipeline runs as a separate Task named "evaluation".
    # It receives the 'args' object which contains our hyperparameters.
    
    # Note: Boolean values in ClearML args usually appear as "True"/"False" strings
    st_val = "True" if softtargets else "False"

    tasks = Task.get_tasks(
        project_name=project_name,
        task_name="evaluation", # The name of the pipeline component
        task_filter={
            "status": ["completed"],
            # Filter by the arguments passed to the evaluation component
            "hyperparams.Args.dataset.value": dataset,
            "hyperparams.Args.mu_algo.value": mu_algo,
            "hyperparams.Args.architecture.value": architecture,
            "hyperparams.Args.softtargets.value": st_val
        }
    )

    if not tasks:
        print("No matching tasks found.")
        return

    print(f"Found {len(tasks)} matching evaluation runs.")
    if len(tasks) < 30:
        print(f"Warning: You have fewer than 30 runs ({len(tasks)}). Statistics might not be robust.")

    # 2. Extract Metrics
    # We will collect 'Accuracy' and 'Loss' scalars
    aggregated_data = {
        "mean_accuracy": [],
        "mean_loss": [],
        # You can add per-class metrics here if needed
    }

    print("Fetching scalar metrics from server...")
    for t in tasks:
        # get_reported_scalars() returns a dict structure: {Title: {Series: {x:[], y:[]}}}
        scalars = t.get_reported_scalars()
        
        if scalars and 'Accuracy' in scalars and 'mean_accuracy' in scalars['Accuracy']:
            # Get the last value reported (y-axis)
            acc = scalars['Accuracy']['mean_accuracy']['y'][-1]
            aggregated_data["mean_accuracy"].append(acc)
            
        if scalars and 'Loss' in scalars and 'mean_loss' in scalars['Loss']:
            loss = scalars['Loss']['mean_loss']['y'][-1]
            aggregated_data["mean_loss"].append(loss)

    # 3. Calculate Statistics
    results_df = pd.DataFrame(aggregated_data)
    
    print("\n--- Aggregated Results ---")
    stats = results_df.describe().loc[['mean', 'std', 'min', 'max']]
    print(stats)
    
    # 4. Visualization (Optional: Box Plot)
    fig = go.Figure()
    fig.add_trace(go.Box(y=results_df['mean_accuracy'], name="Mean Accuracy"))
    fig.update_layout(title=f"Accuracy Distribution over {len(tasks)} runs")
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--mu_algo", type=str, default="graddiff")
    parser.add_argument("--architecture", type=str, default="cnn")
    parser.add_argument("--softtargets", action="store_true")
    parser.add_argument("--project", type=str, default="softtargets")
    
    args = parser.parse_args()
    
    aggregate_runs(args.project, args.dataset, args.mu_algo, args.architecture, args.softtargets)