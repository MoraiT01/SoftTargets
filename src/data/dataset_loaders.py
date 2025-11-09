
import glob
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

# Common image extensions to search for
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.webp']


class SoftTargetDataset(Dataset):
    """
    A PyTorch Dataset that handles the specific directory structure 
    (e.g., root/mnist/0/, root/mnist/7e/) by storing only image paths 
    and loading images on the fly.
    
    The subset_to_include parameter controls which subsets are loaded:
    - 'full': Loads all images from both regular (retained) and 'e' (erased) folders.
    - 'retained': Loads only from regular class folders (0, 1, 2, ...).
    - 'erased': Loads only from the special 'e' folders (0e, 1e, 2e, ...).
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        dataset_name: str,
        subset_to_include: str = 'full', # New parameter
        transform: Optional[Callable] = None,
    ):
        """
        Initializes the dataset by scanning the directory and creating a map 
        of file paths and their true labels.

        Args:
            root_dir: The base directory containing the datasets (e.g., 'data/softtarget_datasets').
            dataset_name: The name of the specific dataset folder (e.g., 'mnist' or 'fashion_mnist').
            subset_to_include: Which subset to load: 'full', 'retained', or 'erased'.
            transform: Optional transform to be applied on a sample.
        """
        super().__init__()
        self.root_path = Path(root_dir) / dataset_name
        self.transform = transform
        self.samples = {}
        
        valid_subsets = ('full', 'retained', 'erased')
        if subset_to_include not in valid_subsets:
            raise ValueError(
                f"Invalid value for subset_to_include: '{subset_to_include}'. "
                f"Must be one of {valid_subsets}"
            )
        self.subset_to_include = subset_to_include

        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_path}")

        print(f"Scanning for images in: {self.root_path} (Mode: {self.subset_to_include})")
        self._find_samples()

        if transform is None:
            print("No transform provided. Using default: Grayscale -> ToTensor.")
            # Default transform to ensure 1-channel greyscale tensor output
            self.transform = T.Compose([
                T.Grayscale(num_output_channels=1), # Guarantees 1-channel output
                T.ToTensor(),                       # Converts PIL Image to PyTorch Tensor
            ])
        else:
            self.transform = transform
        
        if not self.samples:
            print("Warning: No images found. Check your root_dir and subset_to_include setting.")

    def _find_samples(self):
        """
        Scans the directory structure to populate the self.samples list 
        with (image_path, class_label) tuples based on the selected mode.
        """
        # Iterate over all direct subdirectories (class folders)
        for class_dir in self.root_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            dir_name = class_dir.name
            
            # 1. Determine the type of subset
            is_erased_subset = dir_name.endswith('e')
            is_retained_subset = not is_erased_subset
            
            # 2. Filter based on the requested subset_to_include mode
            if self.subset_to_include == 'retained' and is_erased_subset:
                continue
            
            if self.subset_to_include == 'erased' and is_retained_subset:
                continue
                
            # 'full' mode proceeds without skipping
                
            # 3. Determine the true class label
            try:
                if is_erased_subset:
                    # '7e' -> '7' -> 7
                    true_label_str = dir_name[:-1]
                else:
                    # '7' -> 7
                    true_label_str = dir_name
                    
                true_label = int(true_label_str)
            except ValueError:
                # Skip folders that aren't numeric classes (e.g., 'misc', 'README.md')
                continue
            
            # 4. Find all image files within the selected class folder
            for ext in IMAGE_EXTENSIONS:
                for image_path in glob.glob(str(class_dir / ext)):
                    self.samples[len(self.samples)] = (image_path, true_label)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Loads and returns the sample at the given index.
        """
        image_path, label = self.samples[index]

        # 1. Load image from path
        # Open in RGB mode for consistency before transformations, 
        # or rely on a Grayscale transform later if needed.
        image = Image.open(image_path).convert('RGB')
        
        # 2. Apply transformation if provided
        if self.transform:
            image = self.transform(image)
            
        # 3. Convert label to tensor
        target = torch.tensor(label, dtype=torch.long)
        
        return image, target

# -----------------
# Example Usage:
# -----------------
# from torchvision import transforms

# # Define your transformations
# data_transform = transforms.Compose([
#     transforms.Resize((28, 28)), # Example: Resize all to 28x28
#     transforms.Grayscale(num_output_channels=1), # For MNIST/FashionMNIST
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,)) 
# ])

# # 1. Dataset with ONLY the Erased subset
# erased_subset_data = SoftTargetDataset(
#     root_dir="data/softtarget_dataset", 
#     dataset_name="mnist",
#     subset_to_include='erased',
#     transform=data_transform
# )

# # 2. Dataset with ONLY the Retained subset
# retained_subset_data = SoftTargetDataset(
#     root_dir="data/softtarget_dataset", 
#     dataset_name="mnist",
#     subset_to_include='retained',
#     transform=data_transform
# )

# # 3. Dataset with the Full original training data (Retained + Erased)
# full_subset_data = SoftTargetDataset(
#     root_dir="data/softtarget_dataset", 
#     dataset_name="mnist",
#     subset_to_include='full',
#     transform=data_transform
# )