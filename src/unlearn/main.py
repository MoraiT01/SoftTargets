from torch.nn import Module
from typing import Any, Optional
from src.unlearn.algo import gradasc, graddiff, nova
from src.data.dataset_loaders import UnlearningPairDataset, UnlearningDataLoader 

def run_unlearning(
        trained_model: Module,
        unlearn_ds: UnlearningPairDataset,
        algorithm_name: str,
        test_loader: Optional[Any] = None,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.00005,
        optimizer: str = "SGD",
        momentum: float = 0.9,
        alpha: Optional[float] = None,
        noise_samples: Optional[int] = None,
    ) -> Module:
    """
    Selects, configures, and runs the specified unlearning algorithm.
    """
    alg_name_lower = algorithm_name.lower()
    
    unlearn_dl = UnlearningDataLoader(unlearn_ds, batch_size=batch_size, shuffle=True)
    
    if alg_name_lower == "gradasc": # Gradient Ascent
        unlearning_alg = gradasc.GradientAscent(
            model=trained_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
        )
    elif alg_name_lower == "graddiff": # Gradient Difference
        unlearning_alg = graddiff.GradientDifference(
            model=trained_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
            alpha=alpha,
        )
    elif alg_name_lower == "nova": # NOVA
        unlearning_alg = nova.NOVA(
            model=trained_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
            alpha=alpha,
            noise_samples=noise_samples
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    print(f"Running unlearning using {unlearning_alg}...")
    
    # Run the unlearning process
    unlearned_model = unlearning_alg.unlearn(unlearn_dl, test_loader=test_loader)
    
    return unlearned_model