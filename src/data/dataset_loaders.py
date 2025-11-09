import os
import glob
from pathlib import Path
from typing import Callable, Optional, Union, List

import torch
from torch.utils.data import Dataset
from PIL import Image

# Common image extensions to search for
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.webp']


class SoftTargetDataset(Dataset):
    """
    A PyTorch Dataset that handles the specific directory structure 
    (e.g., root/mnist/0/, root/mnist/7e/) by storing only image paths 
    and loading images on the fly.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        dataset_name: str,
        include_erased_subsets: bool = True,
        transform: Optional[Callable] = None,
        image_file_extension: List[str] = ["'*.png'"]
    ):
        """
        Initializes the dataset by scanning the directory and creating a map 
        of file paths and their true labels.

        Args:
            root_dir: The base directory containing the datasets (e.g., 'data/softtarget_datasets').
            dataset_name: The name of the specific dataset folder (e.g., 'mnist' or 'fashion_mnist').
            include_erased_subsets: If True, includes class folders ending in 'e' (erased).
            transform: Optional transform to be applied on a sample.
        """
        super().__init__()
        self.root_path = Path(root_dir) / dataset_name
        self.transform = transform
        self.samples = []
        self.image_file_extension = image_file_extension
        
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_path}")

        print(f"Scanning for images in: {self.root_path}")
        self._find_samples(include_erased_subsets)
        
        if not self.samples:
            print("Warning: No images found. Check your root_dir and file extensions.")

    def _find_samples(self, include_erased_subsets: bool):
        """
        Scans the directory structure to populate the self.samples list 
        with (image_path, class_label) tuples.
        """
        # Iterate over all direct subdirectories (class folders)
        for class_dir in self.root_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            dir_name = class_dir.name
            
            # Check if it's an 'erased' folder (ends in 'e')
            is_erased_subset = dir_name.endswith('e')
            
            if is_erased_subset:
                # If it's an 'erased' folder, extract the true class label (e.g., '7e' -> '7')
                try:
                    true_label_str = dir_name[:-1]
                    true_label = int(true_label_str)
                except ValueError:
                    # Skip folders like 'other_e' or non-numeric classes
                    continue
                    
                # Skip if the user chose not to include erased subsets
                if not include_erased_subsets:
                    continue
            else:
                # It's a normal class folder (e.g., '7')
                try:
                    true_label = int(dir_name)
                except ValueError:
                    # Skip folders like 'misc' or non-numeric classes
                    continue
            
            # Find all image files within the class folder
            for ext in self.image_file_extension:
                for image_path in glob.glob(str(class_dir / ext)):
                    self.samples.append((image_path, true_label))

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
        # Assuming the images are single-channel (e.g., MNIST) or multi-channel
        # Opening as 'L' for grayscale (e.g., MNIST/FashionMNIST) or 'RGB' 
        # is a common choice, let's keep it flexible to the user's transformation.
        # PIL.Image.open is sufficient for most formats.
        image = Image.open(image_path)
        
        # 2. Apply transformation if provided
        if self.transform:
            image = self.transform(image)
            
        # 3. Convert label to tensor
        target = torch.tensor(label, dtype=torch.long)
        
        return image, target

# --- Example Usage ---
# from torchvision import transforms

# # Define your transformations
# data_transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1), # For MNIST/FashionMNIST
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,)) # Example normalization
# ])

# # Example of creating a dataset *including* the erased subset '7e'
# full_mnist_data = SoftTargetDataset(
#     root_dir="data/softtarget_dataset", # Adjust this path as needed
#     dataset_name="mnist",
#     include_erased_subsets=True,
#     transform=data_transform
# )

# # Example of creating a dataset *excluding* the erased subset '7e'
# retained_mnist_data = SoftTargetDataset(
#     root_dir="data/softtarget_dataset", # Adjust this path as needed
#     dataset_name="mnist",
#     include_erased_subsets=False,
#     transform=data_transform
# )

# # print(f"Total samples (with erased): {len(full_mnist_data)}")
# # print(f"Total samples (retained only): {len(retained_mnist_data)}")