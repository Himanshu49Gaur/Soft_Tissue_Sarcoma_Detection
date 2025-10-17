"""
# Dataset Loading and Initial Exploration
## Objective:
Inspect and verify the directory structure of the Osteosarcoma dataset.

## Details:
- Iterates through the dataset folder hierarchy.
- Displays each directory, its subdirectories, and the number of files.
- Helps confirm proper dataset organization before preprocessing or training.
"""

import os
import pandas as pd

dataset_path = "path_to_your_dataset_directory"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
else:
    print(f"Dataset directory located at: {dataset_path}\n")

print("Checking dataset structure...\n")
for root, dirs, files in os.walk(dataset_path):
    print(f"Directory: {root}")
    print(f"Subdirectories: {dirs}")
    print(f"Number of Files: {len(files)}\n")

print("Dataset structure exploration completed successfully.")
