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

# 3.1 Class Distribution After Preprocessing
plt.figure(figsize=(8, 5))
sns.countplot(x=y_train, palette="viridis")
plt.xticks(ticks=[0, 1, 2], labels=class_labels)
plt.title("Class Distribution in Training Set After Augmentation")
plt.xlabel("Tumor Class")
plt.ylabel("Number of Images")
plt.show()

# 3.2 Visualizing Preprocessed Images
fig, axes = plt.subplots(3, 5, figsize=(15, 8))
fig.suptitle("Sample Augmented Images from Training Set", fontsize=14)
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_train))
    ax.imshow(X_train[idx])
    ax.set_title(class_labels[y_train[idx]])
    ax.axis("off")
plt.show()
