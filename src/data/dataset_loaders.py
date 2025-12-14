import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
# Import default_collate for clean batching of individual lists
from torch.utils.data.dataloader import default_collate 
from torchvision import transforms
from PIL import Image
import os
import random
from typing import Tuple, Any, Dict, List, Optional, Literal
# --- Configuration Constants ---
# after running the indexing script.
CSV_FILE_PATH = "data/mnist_index.csv" 

# if the paths in the CSV are relative. If the paths are absolute, set this to "" or "."
IMAGE_ROOT_DIR = "."

# Define standard transformations for the images
DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),         # Convert PIL Image to a PyTorch Tensor
])

# --- Base Dataset Class ---

class BaseDataset(Dataset):
    """
    Parent class for all datasets. Handles CSV loading, image loading,
    and common helper functions.
    """
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        Initializes the dataset by loading the CSV file.
        
        Args:
            csv_file (str): Path to the dataset index CSV file.
            root_dir (str): Root directory for image files (used to resolve relative paths).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at: {csv_file}")

        print(f"Loading dataset index from {csv_file}...")
        self.data_frame = pd.read_csv(csv_file, sep=';')
        self.root_dir = root_dir
        self.transform = transform if transform is not None else DEFAULT_TRANSFORMS
        np_tensor_to_label = {cls: str(i) for i, cls in enumerate(self.data_frame['Class_Label'].unique())}
        self.tensor_to_label = {}
        # Convert class labels to one-hot vectors
        for key in np_tensor_to_label.keys():
            zeroer = torch.zeros(len(np_tensor_to_label))
            index = np_tensor_to_label[key]
            zeroer[int(index)] = 1
            self.tensor_to_label[zeroer] = index
        self.label_to_tensor = {i: cls for cls, i in self.tensor_to_label.items()}

    def _load_image(self, img_path: str) -> torch.Tensor:
        """Helper to load and transform an image."""
        # full_path = os.path.join(self.root_dir, img_path)
        try:
            # CHANGE: Convert to 'L' (Grayscale) instead of 'RGB'
            image = Image.open(img_path).convert('L') #
        except Exception as e:
            print(f"Error loading image at {img_path}: {e}")
            # Return a zero tensor or handle error appropriately
            return torch.zeros(1, 28, 28)
        
        if self.transform:
            return self.transform(image) # type: ignore
        return transforms.ToTensor()(image)

    def _get_label_tensor(self, class_label: str) -> torch.Tensor:
        """Helper to convert a class label string to a PyTorch LongTensor."""
        tensor = self.label_to_tensor.get(class_label)
        if tensor is None:
            raise ValueError(f"Unknown class label: {class_label}")
        return tensor.detach().clone().requires_grad_(True)
        
    def __len__(self) -> int:
        """This should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement __len__.")

    def __getitem__(self, idx: int) -> Any:
        """This should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement __getitem__.")

# --- 1. Standard Training/Testing Dataset Class ---

class TrainTestDataset(BaseDataset):
    """
    A standard PyTorch dataset for training or testing a model.
    Filters the data based on the 'train_split' column.
Extended to support sampling modes (all/forget/retain), class filtering, 
    and dynamic forget column selection.
    """
    def __init__(self, 
                 csv_file: str, 
                 root_dir: str, 
                 split: str = 'train', 
                 sample_mode: Literal["all", "forget", "retain"] = "all",
                 classes: Optional[List[str]] = None,
                 forget_split_col: str = "f1_split",
                 transform=None):
        """
        Args:
            csv_file (str): Path to the CSV.
            root_dir (str): Root directory for images.
            split (str): 'train' (train_split=1) or 'test' (train_split=0).
            sample_mode (str): 'all', 'forget' (only forget samples), or 'retain' (only non-forget samples).
            classes (List[str]): List of class labels to include. If None or empty, includes all.
            forget_split_col (str): The column name in the CSV defining the forget split (1=Forget, 0=Retain).
            transform: Optional transform.
        """
        super().__init__(csv_file, root_dir, transform)
        
        # 1. Determine the filter condition based on the desired split
        if split.lower() == 'train':
            split_filter = 1
        elif split.lower() == 'test':
            split_filter = 0
        else:
            raise ValueError("Split must be 'train' (train_split=1) or 'test' (train_split=0).")

        # Initial filtering by Train/Test Split
        # NOTE: Assumes 'train_split' column is present in the CSV
        subset_df = self.data_frame[self.data_frame['train_split'] == split_filter]

        # 2. Filter by Classes (if provided)
        if classes:
            # Convert column to string to ensure matching with the provided string list
            subset_df = subset_df[subset_df['Class_Label'].astype(str).isin(classes)]

        # 3. Filter by Sample Mode (Forget vs Retain)
        if sample_mode != "all":
            if forget_split_col not in subset_df.columns:
                raise ValueError(f"Forget split column '{forget_split_col}' not found in CSV.")
            
            if sample_mode == "forget":
                # Keep only rows where the forget column is 1
                subset_df = subset_df[subset_df[forget_split_col] == 1]
            elif sample_mode == "retain":
                # Keep only rows where the forget column is 0
                subset_df = subset_df[subset_df[forget_split_col] == 0]
            else:
                raise ValueError(f"Invalid sample_mode: {sample_mode}. Must be 'all', 'forget', or 'retain'.")

        self.subset_df = subset_df.reset_index(drop=True)
        print(f"Loaded {split} split (mode='{sample_mode}', classes={classes}) with {len(self.subset_df)} samples.")

    def __len__(self) -> int:
        """Returns the total number of samples in the filtered subset."""
        return len(self.subset_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and returns a single data sample (image tensor and label tensor).
        """
        if idx >= len(self.subset_df):
            raise IndexError("Index out of bounds for the current dataset subset.")
            
        # Get row data for the requested index
        img_path = str(self.subset_df.loc[idx, 'Path'])
        class_label = str(self.subset_df.loc[idx, 'Class_Label'])

        # Load image and get label tensor
        image_tensor = self._load_image(img_path)
        label_tensor = self._get_label_tensor(class_label)

        return image_tensor, label_tensor

    def get_tags(self, idx: int) -> Dict:
        if idx >= len(self.subset_df):
            raise IndexError("Index out of bounds for the current dataset subset.")
        return self.subset_df.loc[idx].to_dict()

# --- 2. Unlearning Pair Dataset Class ---

class UnlearningPairDataset(BaseDataset):
    """
    Dataset designed for machine unlearning. It returns a paired sample:
    (Forget sample, Non-Forget sample).
    
    It now supports a train/test split based on the 'train_split' column.
    Length is based on the number of forget samples (f1_split=1) within the chosen split.
    """
    def __init__(self, 
                 csv_file: str, 
                 root_dir: str, 
                 split: str = 'train', 
                 forget_split_col: str = "f1_split", 
                 transform=None):
        super().__init__(csv_file, root_dir, transform)
        
        # 1. Determine the filter condition based on the desired train/test split
        if split.lower() == 'train':
            filter_condition = 1
        elif split.lower() == 'test':
            filter_condition = 0
        else:
            raise ValueError("Split must be 'train' (train_split=1) or 'test' (train_split=0).")

        # 2. Filter the base DataFrame by train/test split first
        filtered_df = self.data_frame[self.data_frame['train_split'] == filter_condition]

        # Check if the requested forget column exists
        if forget_split_col not in filtered_df.columns:
            raise ValueError(f"Forget split column '{forget_split_col}' not found in CSV.")

        # 3. Separate the filtered DataFrame into Forget and Non-Forget sets based on the forget column
        # Uses the dynamic 'forget_split_col': 1 for Forget, 0 for Non-Forget
        self.forget_df = filtered_df[filtered_df[forget_split_col] == 1].reset_index(drop=True)
        self.non_forget_df = filtered_df[filtered_df[forget_split_col] == 0].reset_index(drop=True)

        if len(self.forget_df) == 0:
            raise ValueError(f"No Forget ({forget_split_col}=1) samples found in the '{split}' split.")
        if len(self.non_forget_df) == 0:
            raise ValueError(f"No Non-Forget ({forget_split_col}=0) samples found in the '{split}' split.")

        print(f"Loaded Unlearning Dataset ({split} split using '{forget_split_col}'): {len(self.forget_df)} Forget samples and {len(self.non_forget_df)} Non-Forget samples.")

        # Storage for soft targets
        self.use_soft_targets = False
        self.soft_targets_forget: Dict[int, torch.Tensor] = {}
        self.soft_targets_retain: Dict[int, torch.Tensor] = {}

    def make_softtargets(self, model: nn.Module, device: torch.device):
        """
        Parses the model to generate soft targets (predicted probabilities) 
        for all samples in the dataset.
        """
        self.use_soft_targets = True
        model.eval()
        model.to(device)

        print("Generating soft targets for Forget Set...")
        for i in range(len(self.forget_df)):
            row = self.forget_df.loc[i]
            img_path = str(row['Path'])
            # Load and prepare image (add batch dim)
            img_tensor = self._load_image(img_path).unsqueeze(0).to(device)

            with torch.no_grad():
                # Model outputs log_softmax, so we apply exp to get probabilities
                output = model(img_tensor)
                probs = torch.exp(output).squeeze(0).cpu()
                
            self.soft_targets_forget[i] = probs

        print("Generating soft targets for Retain Set...")
        for i in range(len(self.non_forget_df)):
            row = self.non_forget_df.loc[i]
            img_path = str(row['Path'])
            img_tensor = self._load_image(img_path).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.exp(output).squeeze(0).cpu()

            self.soft_targets_retain[i] = probs
            
        print("Soft targets generated successfully.")

    def __len__(self) -> int:
        """
        The length of the dataset is equal to the number of Forget samples,
        as each index will yield one Forget-Non-Forget pair.
        """
        return len(self.forget_df)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Loads and returns a paired sample: (Forget_Sample, Non_Forget_Sample).
        Each sample is returned as a dictionary containing its tensor and label.
        """
        if idx >= len(self.forget_df):
            raise IndexError("Index out of bounds for the current dataset subset.")

        # 1. Get the Forget Sample Data
        forget_row = self.forget_df.loc[idx]
        f_img_path = str(forget_row['Path'])
        f_class_label = str(forget_row['Class_Label'])
        
        # 2. Get the Non-Forget Sample Data
        # Use modulo arithmetic to wrap around the non-forget set
        
        # nf_idx = idx % len(self.non_forget_df) # This might bring some bias into it
        # NOTE: Maybe random sampling could also work nicely
        nf_idx = random.randint(0, len(self.non_forget_df) - 1)
        non_forget_row = self.non_forget_df.loc[nf_idx]
        nf_img_path = str(non_forget_row['Path'])
        nf_class_label = str(non_forget_row['Class_Label'])

        # 3. Load Tensors
        f_image_tensor = self._load_image(f_img_path)
        nf_image_tensor = self._load_image(nf_img_path)

        # 4. Get Labels (Soft or Hard)
        if self.use_soft_targets:
            # Use stored soft targets
            # Note: We detach/clone to be safe, though they are already on CPU
            f_label_tensor = self.soft_targets_forget[idx].detach().clone()
            nf_label_tensor = self.soft_targets_retain[nf_idx].detach().clone()
        else:
            # Use standard one-hot hard labels
            f_label_tensor = self._get_label_tensor(f_class_label)
            nf_label_tensor = self._get_label_tensor(nf_class_label)

        # 5. Return as a pair of dictionaries for clarity
        forget_sample = {
            'image': f_image_tensor,
            'label': f_label_tensor,
            'original_index': forget_row['index'],
            'path': f_img_path
        }
        
        non_forget_sample = {
            'image': nf_image_tensor,
            'label': nf_label_tensor,
            'original_index': non_forget_row['index'],
            'path': nf_img_path
        }

        return forget_sample, non_forget_sample

# --- 3. Custom Collation Function for Unlearning Data ---

def unlearning_collate_fn(batch: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Custom collation function for UnlearningPairDataset.
    
    It takes a list of (forget_sample, non_forget_sample) pairs and returns 
    a single dictionary containing two structured batches (forget and retain).
    
    Args:
        batch (List): A list of paired dictionaries, where each pair is 
                      (forget_dict, non_forget_dict).
                      
    Returns:
        Dict: A dictionary structured as:
              {
                "forget": {"input": <tensor_batch>, "labels": <tensor_batch>},
                "retain": {"input": <tensor_batch>, "labels": <tensor_batch>}
              }
    """
    # 1. Separate the forget and non-forget samples into two lists
    forget_list = [item[0] for item in batch]
    retain_list = [item[1] for item in batch]
    
    # 2. Use the standard collate function to batch each list of dictionaries separately
    forget_batch = default_collate(forget_list)
    retain_batch = default_collate(retain_list)
    
    # 3. Restructure and rename keys for the final output format
    
    # Process Forget Batch
    forget_output = {
        "input": forget_batch.pop('image'),
        "labels": forget_batch.pop('label'),
        # Optional metadata (will be batched as lists/tensors depending on type)
        "metadata": forget_batch 
    }
    
    # Process Retain (Non-Forget) Batch
    retain_output = {
        "input": retain_batch.pop('image'),
        "labels": retain_batch.pop('label'),
        # Optional metadata
        "metadata": retain_batch
    }
    
    return {
        "forget": forget_output,
        "retain": retain_output
    }

# --- 4. Custom DataLoader Class ---

class UnlearningDataLoader(DataLoader):
    """
    A specialized DataLoader that automatically uses the custom collate_fn 
    for UnlearningPairDataset, simplifying the batching process.
    """
    def __init__(self, dataset: UnlearningPairDataset, **kwargs):
        """
        Initializes the DataLoader, ensuring the correct collate_fn is used.
        """
        if not isinstance(dataset, UnlearningPairDataset):
            raise TypeError("UnlearningDataLoader is designed only for UnlearningPairDataset.")

        # Set the custom collation function and pass all other arguments to the parent class
        super().__init__(dataset, collate_fn=unlearning_collate_fn, **kwargs)
