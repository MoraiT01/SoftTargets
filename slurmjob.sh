#!/bin/bash
#SBATCH --job-name=softtargets # specify the job name for monitoring
#SBATCH --output=transformer-out/softtargets_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/softtargets_JOB_%j.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=4 # Number of CPUs
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.39gb:1  # Allocate 1/2 GPU resources with specified configurations
#SBATCH --mem=50G  # Specify the total amount of memory
#SBATCH --time=25:00:00  # Set a time limit for the job
#SBATCH --partition=advance
#SBATCH --qos=advance
#SBATCH --account=advance

# This file serves as a wrapper, if we want to execute the process on the HPC using SLURM

# Run the Python script
srun hostname

### \/\/\/\/ ###
# # This should be used, if we can not run the project inside a docker image
# # Initialize Conda for the current shell session
# # Replace '~/miniconda3' with the actual path to your Miniconda/Anaconda installation if different
source /fast_storage/kastler/miniconda3/etc/profile.d/conda.sh

ENV_NAME="softtargets"
ENV_PATH="/fast_storage/kastler/miniconda3/envs/${ENV_NAME}"

# --- OPTIMIZATION: Check for Environment Existence ---

# Check if the environment directory exists
if [ -d "${ENV_PATH}" ]; then
    echo "Conda environment '${ENV_NAME}' already exists. Activating and updating dependencies..."
    
    # 1. Activate the existing environment
    conda activate "${ENV_NAME}"
    
    # 2. Update dependencies (This is much faster than a full reinstall)
    pip install -r requirements.txt
    
else
    echo "Conda environment '${ENV_NAME}' does not exist. Creating it now..."
    
    # 1. Create the environment with Python 3.11
    # Use mamba if available for faster solve times, otherwise use conda
    conda create -n "${ENV_NAME}" python=3.11 -y 
    
    # 2. Activate the newly created environment
    conda activate "${ENV_NAME}"
    
    # 3. Install all dependencies
    pip install -r requirements.txt
fi

# --- VERIFICATION ---
echo "--- Environment Status ---"
python --version
pip freeze | grep -Ff requirements.txt # Check if key packages are installed

# Final activation command (often redundant after the block, but ensures activation)
# Note: You should only need one of the activation commands, 
# either 'conda activate posthoc' or 'conda activate /path/to/env'
conda activate "${ENV_NAME}"

### /\/\/\/\ ###

# To make sure, we can talk to the ClearML server, we set the config file path here
export CLEARML_CONFIG_FILE="/fast_storage/kastler/clearml.conf"

python -c "import torch; print(torch.cuda.is_available())"

### Now you may start your operations below ###

# The pipeline can be run multiple times
# Hyperparameters are:
# - dataset: mnist, fashion_mnist
# - mu_algo: gradasc, graddiff
# - architecture: mlp, cnn
# - softtargets
# Example Call: python main.py --dataset mnist --mu_algo gradasc --architecture mlp --softtargets

# DATASET="fashion_mnist"
# MU_ALGO="graddiff"
# ARCHITECTURE="mlp"
# SOFTTARGETS=false 

### Main Loops ###
# We need to loop over every permutation of the hyperparameters
# Here we only run one configuration for demonstration purposes

# Loop over dataset
for DATASET in "mnist" "fashion_mnist"
do
  # loop over mu_algo
  for MU_ALGO in "gradasc" "graddiff" "nova"
  do
    # loop over architecture
    for ARCHITECTURE in "mlp" "cnn"
    do
      # loog over softtargets
      for SOFTTARGETS in true false
      do
        echo "Running configuration: Dataset=$DATASET, MU_ALGO=$MU_ALGO, ARCHITECTURE=$ARCHITECTURE, SOFTTARGETS=$SOFTTARGETS"

        ### Inner Loop: Multiple Runs for Evaluation ###
        # For evaluation purposes, we run it a minimul of 30 times
        # for (( i=1; i<=30; i++ ))
        # do
          echo "Running iteration: $i"

        if $SOFTTARGETS; then
          python main.py --dataset $DATASET --mu_algo $MU_ALGO --architecture $ARCHITECTURE --softtargets
        else
          python main.py --dataset $DATASET --mu_algo $MU_ALGO --architecture $ARCHITECTURE
        fi
        # done # End of 30runs

        ### Final Aggregation Step ###
        # python src/eval/aggregate_results.py --dataset $DATASET --mu_algo $MU_ALGO --architecture $ARCHITECTURE --softtargets $SOFTTARGETS

      done # End of softtargets
    done # End of architecture
  done # End of mu_algo
done # End of dataset

echo "All configurations have been executed."