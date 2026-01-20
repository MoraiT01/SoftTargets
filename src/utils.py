from clearml import Task

def load_model_from_task(task_id: str, artifact_name: str = "trained Model"):
    """
    Connects to a ClearML Task and retrieves a pickled model artifact (e.g. from a Pipeline step).
    
    Args:
        task_id (str): The ID of the specific task (e.g. the "Model Training" step).
        artifact_name (str): The name of the artifact to retrieve. Defaults to "trained Model" 
                             (matches 'return_values' in the pipeline component).

    Returns:
        The deserialized model object (e.g. LitResNet).
    """
    print(f"Connecting to task: {task_id}")
    task = Task.get_task(task_id=task_id)
    
    if artifact_name in task.artifacts:
        print(f"Found artifact: '{artifact_name}'. Downloading and deserializing...")
        # .get() downloads the pickle and returns the Python object
        return task.artifacts[artifact_name].get()
    else:
        available_artifacts = list(task.artifacts.keys())
        raise ValueError(
            f"Artifact '{artifact_name}' not found in task {task_id}.\n"
            f"Available artifacts: {available_artifacts}"
        )