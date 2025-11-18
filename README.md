# SoftTargets
This Repository aims to show the differences between using hard and soft labels for Machine Unlearning Algorithms, like Gradient Ascent, Gradient Difference, RMU and Tarun, Ayush K., et al. "Fast yet Efficient Unlearning Machine Unlearning". 

## Setup

This project is managed using Docker.

1.  **Build the Docker image:**
    ```bash
    docker build -t softtargets .
    ```

2.  **Run the container:**
    ```bash
    docker run it --rm -v ./data:/app/data softtargets
    ```

    For developement, clearML is used.
    If you want to utilize it as well, use the following command:
    ```
    docker run -it --rm -v ./data:/app/data -v "</path/to/your/clearml.conf>":/root/clearml.conf softtargets
    ```

## Configurations
...
