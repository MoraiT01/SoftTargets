"""Handling HP Optimization specifically for the Unlearning Step of the Pipeline"""

from src.utils import load_model_from_task

# The chosen Task were successfully executed before
baseline    = load_model_from_task("3cebe0e4059e4fa58c57bd9a650ef7f5", "Baseline Model")
unlearn_ds  = load_model_from_task("ce89c3a50dda4ef1809314d2bce71374", "Test Dataloader")
test_dl     = load_model_from_task("ce89c3a50dda4ef1809314d2bce71374", "Unlearning Dataset")

from clearml.automation import (
    UniformParameterRange,
    UniformIntegerParameterRange,
    DiscreteParameterRange,
    HyperParameterOptimizer
)
from clearml import Task

Task.init(
    project_name="softtargets",
    task_name=f"optimizer",
    task_type=Task.TaskTypes.optimizer,
)

optimizer = HyperParameterOptimizer(

    base_task_id="",
    hyper_parameters=[

    ],
    objective_metric_title="Accuracy Difference",
    objective_metric_series="Test: Accuracy",
    objective_metric_sign="min"
)

optimizer.start()
optimizer.wait()
optimizer.stop()