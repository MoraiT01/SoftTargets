from torch.nn import Module
from typing import Any, Optional, Literal
from src.unlearn.algo import gradasc, graddiff, nova
from src.data.dataset_loaders import UnlearningPairDataset, UnlearningDataLoader 

def run_unlearning(
        trained_model: Module,
        unlearn_ds: UnlearningPairDataset,
        algorithm_name: Literal["gradasc", "graddiff", "nova"],
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
    unlearn_dl = UnlearningDataLoader(unlearn_ds, batch_size=batch_size, shuffle=True)
    
    if str(algorithm_name) == "gradasc": # Gradient Ascent
        unlearning_alg = gradasc.GradientAscent(
            model=trained_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
        )
    elif str(algorithm_name) == "graddiff": # Gradient Difference
        unlearning_alg = graddiff.GradientDifference(
            model=trained_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
            alpha=alpha,
        )
    elif str(algorithm_name) == "nova": # NOVA
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