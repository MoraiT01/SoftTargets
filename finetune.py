from src.utils import load_model_from_task
from clearml import Task
from clearml.automation import (
    UniformParameterRange,
    DiscreteParameterRange,
    HyperParameterOptimizer
)

# Note: While you load these artifacts here, the Optimizer works by cloning a Base Task.
# Ensure your Base Task is configured to use these specific artifacts (e.g., via fixed arguments)
# or that the Base Task logic dynamically loads the correct files.
# baseline    = load_model_from_task("3cebe0e4059e4fa58c57bd9a650ef7f5", "Baseline Model")
# unlearn_ds  = load_model_from_task("ce89c3a50dda4ef1809314d2bce71374", "Test Dataloader")
# test_dl     = load_model_from_task("ce89c3a50dda4ef1809314d2bce71374", "Unlearning Dataset")

Task.init(
    project_name="softtargets",
    task_name="optimizer",
    task_type=Task.TaskTypes.optimizer,
)

optimizer = HyperParameterOptimizer(
    # [REQUIRED] ID of the template task to clone. 
    # This task should be a successful run of the unlearning script.
    base_task_id="INSERT_YOUR_BASE_TASK_ID_HERE", 
    
    # [REQUIRED] Define the search space
    hyper_parameters=[
        # Example: Tuning the learning rate (match the argument name in your base task)
        # If your base task uses argparse, it might be 'Args/learning_rate'
        # If using a config dictionary connected to ClearML, it might be 'General/unlearning.learning_rate'
        UniformParameterRange(
            name='kwargs/learning_rate', 
            min_value=0.00001, 
            max_value=0.001, 
            step_size=0.00001
        ),
        # Example: Tuning Alpha for Gradient Difference
        # For GD und NOVA
        UniformParameterRange(
            name='kwargs/alpha', 
            min_value=0.1, 
            max_value=5.0, 
            step_size=0.1
        ),
        # Only for NOVA
        DiscreteParameterRange(
            name='kwargs/noise_samples', 
            values=[1, 2, 4, 8, 16, 32]
        ),

    ],
    
    # [CHECK] Metric to optimize. 
    # Based on src/unlearn/algo/graddiff.py, Test Accuracy is logged as:
    # title="Metric Changes (GD)", series="Test Accuracy" # For Gradient Difference
    # title="Metric Changes (GA)", series="Test Accuracy" # For Gradient Ascent
    # title="Metric Changes (NOVA)", series="Test Accuracy" # For NOVA
    objective_metric_title="Unlearning (NOVA)", 
    objective_metric_series="Test Accuracy",
    objective_metric_sign="min", # We generally want to Maximize accuracy (or minimize loss)
    
    # Strategy settings
    max_number_of_concurrent_tasks=2,
    optimizer_class='OptimizerRandomSearch', # Or OptimizerOptuna for bayesian optimization
    execution_queue='default', # Ensure you have an agent listening to this queue
    time_limit_per_job=30.0, # minutes
    pool_period_min=0.2,
    total_max_jobs=50,
)

# Start the optimization
# Use job_complete_callback=None to avoid blocking locally if running on agents
optimizer.start(job_complete_callback=None) 

# Wait for completion
optimizer.wait()

# Stop the optimizer
optimizer.stop()