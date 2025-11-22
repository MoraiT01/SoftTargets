import os
import pandas as pd
from utils import is_ignored

DATASETS = ["mnist", "cmnist", "fashion_mnist"]
# **Change 'path/to/your/dataset/root_folder' to the actual path!**
ROOT_FOLDER = "data" 

def create_ml_csv(root_dir, output_csv_path="dataset_index.csv"):
    """
    Creates a CSV file indexing image files from a folder structure.

    Args:
        root_dir (str): The path to the main folder containing class subfolders.
        output_csv_path (str): The filename for the output CSV.
    """
    data = []
    
    # Traverse the directory
    # os.walk yields (dirpath, dirnames, filenames)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Extract the class label from the last part of the path (the folder name)
        # This assumes the subfolders directly under root_dir are the class names.
        # We skip the root directory itself by checking if it's not the root_dir
        if dirpath != root_dir:
            class_label = os.path.basename(dirpath)
        else:
            # Skip files that might be in the root directory itself
            continue 

        # Process files in the current directory
        for filename in filenames:
            # Check if the file is a .png image (case-insensitive)
            if filename.lower().endswith('.png'):
                # 1. Full path to the file
                file_path = os.path.join(dirpath, filename)
                
                # 2. Append the structured data
                # The index will be added automatically by Pandas later
                # Forget tag is initialized to 0
                if "e" in class_label:
                    data.append({
                        'Path': file_path.replace("\\", "/"),
                        'Class_Label': class_label.replace("e", ""),
                        'f1_split': 1 
                    })
                else:
                    data.append({
                        'Path': file_path.replace("\\", "/"),
                        'Class_Label':class_label,
                        'f1_split': 0 
                    })

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Add the 'index' column as the sequential index of the DataFrame
    df.insert(0, 'index', range(len(df)))

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False, encoding='utf-8', sep=';')
    
    print(f"✅ Successfully created {output_csv_path} with {len(df)} entries.")
    print(f"Columns: {list(df.columns)}")

    with open(f"{ROOT_FOLDER}/.gitignore", 'a') as f:
        pattern = f"{output_csv_path.split('/')[-1]}"
        if is_ignored(pattern, f'{ROOT_FOLDER}/.gitignore'):
            print(f"{pattern} is already in the .gitignore file.")
        else:
            f.write(f"{pattern}\n")

# --- Configuration ---
# Set the path to the main directory that contains your class folders
# Example structure:
# root_folder/
# ├── cat/
# │   ├── 001.png
# │   └── 002.png
# └── dog/
#     ├── 003.png
#     └── 004.png
# ---------------------

# Run the function
for name in DATASETS:
    create_ml_csv(os.path.join(ROOT_FOLDER, "softtarget_dataset", name), f"data/{name}_index.csv") 

# --- Example Output Structure ---
# If you ran the script, the CSV would look like this:
# index,Path,Class_Label,Forget_Tag
# 0,path/to/your/dataset/root_folder/cat/001.png,cat,0
# 1,path/to/your/dataset/root_folder/cat/002.png,cat,0
# 2,path/to/your/dataset/root_folder/dog/003.png,dog,0
# 3,path/to/your/dataset/root_folder/dog/004.png,dog,0
# -------------------------------
