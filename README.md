# SoftTargets

This repository demonstrates the differences between using **Hard Labels** and **Soft Labels** (probability distributions) in Machine Unlearning algorithms. It implements and compares methods such as Gradient Ascent and Gradient Difference on datasets like MNIST and Fashion-MNIST.

## 🚀 Execution

There are three ways to run this project:

### 1. Docker (Recommended)

The container provides an isolated environment.

**Build:**

```bash
docker build -t softtargets .

```

**Run:**
Mounts the local `data` directory to persist datasets. Optionally, a ClearML configuration can be mounted.

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v "/path/to/your/clearml.conf":/root/clearml.conf \
  softtargets

```

### 2. Local (Python / Conda)

Prerequisite: Python 3.11.

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python main.py --dataset mnist --mu_algo gradasc --architecture mlp

```

### 3. HPC (SLURM)

A batch script is provided for execution on a cluster, which also handles the environment setup.

```bash
sbatch slurmjob.sh

```

## ⚙️ Usage & CLI Arguments

The main pipeline is controlled via `main.py`. The following arguments are available:

| Argument | Description | Options |
| --- | --- | --- |
| `--dataset` | Dataset to use | `mnist`, `fashion_mnist` |
| `--mu_algo` | Unlearning Algorithm | `gradasc`, `graddiff` |
| `--architecture` | Model Architecture | `mlp`, `cnn` |
| `--softtargets` | Flag for Soft Labels | (Set flag for `True`, otherwise `False`) |

**Example:**

```bash
python main.py --dataset fashion_mnist --mu_algo graddiff --architecture cnn --softtargets

```

## 🛠 Configuration

Detailed hyperparameters (learning rate, epochs, batch size) are managed via YAML files and do not need to be changed in the code:

**Training (Model-Specific):**
* Path: `configs/training/`
* Files: `cnn.yaml`, `mlp.yaml`


**Unlearning (Algorithm-Specific):**
* Path: `configs/unlearn/`
* Files: `gradasc.yaml`, `graddiff.yaml`, `RMU.yaml`
