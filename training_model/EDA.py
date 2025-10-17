import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set preprocessed data directory (edit as needed)
preprocessed_dir = "path_to_preprocessed_data"  # Example: "./preprocessed_osteosarcoma"

# Verify preprocessed directory
if not os.path.exists(preprocessed_dir):
    raise FileNotFoundError(f"Preprocessed directory not found at {preprocessed_dir}. Please run preprocessing first.")

# Load preprocessed datasets
X_train = np.load(os.path.join(preprocessed_dir, "X_train_tensor.npy"))
y_train = np.load(os.path.join(preprocessed_dir, "y_train_tensor.npy"))
X_val = np.load(os.path.join(preprocessed_dir, "X_validate_tensor.npy"))
y_val = np.load(os.path.join(preprocessed_dir, "y_validate_tensor.npy"))
X_test = np.load(os.path.join(preprocessed_dir, "X_test_tensor.npy"))
y_test = np.load(os.path.join(preprocessed_dir, "y_test_tensor.npy"))

# Define class labels
class_labels = ["Non-Tumor", "Non-Viable-Tumor", "Viable"]
