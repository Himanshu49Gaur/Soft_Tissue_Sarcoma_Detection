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

# 3.3 Pixel Intensity Distribution (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(X_train.flatten(), bins=50, kde=True, color='purple')
plt.title("Pixel Intensity Distribution in Training Set")
plt.xlabel("Pixel Intensity (Normalized)")
plt.ylabel("Frequency")
plt.show()

# 3.4 Dataset Shape & Validation
dataset_info = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Image Shape": [X_train.shape[1:], X_val.shape[1:], X_test.shape[1:]],
    "Total Images": [X_train.shape[0], X_val.shape[0], X_test.shape[0]],
    "Classes": [len(set(y_train)), len(set(y_val)), len(set(y_test))],
})
print("Dataset Overview:")
print(dataset_info.to_string(index=False))

# 3.5 Mean and Standard Deviation of Pixel Intensities
mean_intensity = np.mean(X_train)
std_intensity = np.std(X_train)
print(f"\nMean Pixel Intensity: {mean_intensity:.4f}")
print(f"Standard Deviation of Pixel Intensities: {std_intensity:.4f}")

# 3.6 Correlation Between Image Classes (One-Hot Encoding)
y_train_one_hot = pd.get_dummies(y_train)
correlation_matrix = y_train_one_hot.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Correlation Matrix Between Classes")
plt.show()

# 3.7 Check for Class Imbalance (Pie Chart)
unique_labels, label_counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(6, 6))
plt.pie(label_counts, labels=class_labels, autopct="%1.1f%%",
        colors=["gold", "lightcoral", "lightskyblue"])
plt.title("Class Distribution in Training Set")
plt.show()

# 3.8 Pixel Intensity Comparison Between Classes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, class_label in enumerate(class_labels):
    class_indices = np.where(y_train == i)[0]
    class_pixels = X_train[class_indices].flatten()
    sns.histplot(class_pixels, bins=50, kde=True, ax=axes[i], color=np.random.rand(3,))
    axes[i].set_title(f"Pixel Intensity Distribution ({class_label})")
    axes[i].set_xlabel("Pixel Intensity")
    axes[i].set_ylabel("Frequency")
plt.show()

# 3.9 Brightness Distribution
brightness_values = {class_name: [] for class_name in class_labels}
for i, class_name in enumerate(class_labels):
    class_indices = np.where(y_train == i)[0]
    sample_images = np.random.choice(class_indices, size=min(10, len(class_indices)), replace=False)
    for idx in sample_images:
        brightness_values[class_name].append(np.mean(X_train[idx]))
df_brightness = pd.DataFrame(brightness_values)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_brightness)
plt.title("Brightness Distribution Across Classes")
plt.ylabel("Mean Pixel Value")
plt.xlabel("Tumor Class")
plt.xticks(ticks=[0, 1, 2], labels=class_labels)
plt.show()

# 3.10 Aspect Ratio Analysis
aspect_ratios = {class_name: [] for class_name in class_labels}
for i, class_name in enumerate(class_labels):
    class_indices = np.where(y_train == i)[0]
    sample_images = np.random.choice(class_indices, size=min(10, len(class_indices)), replace=False)
    for idx in sample_images:
        h, w, _ = X_train[idx].shape
        aspect_ratios[class_name].append(w / h)
df_aspect_ratios = pd.DataFrame(aspect_ratios)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_aspect_ratios)
plt.title("Aspect Ratio Distribution Across Classes")
plt.ylabel("Width / Height Ratio")
plt.xlabel("Tumor Class")
plt.xticks(ticks=[0, 1, 2], labels=class_labels)
plt.show()

# 3.11 RGB Channel Mean Analysis
channel_means = {class_name: [] for class_name in class_labels}
for i, class_name in enumerate(class_labels):
    class_indices = np.where(y_train == i)[0]
    sample_images = np.random.choice(class_indices, size=min(10, len(class_indices)), replace=False)
    for idx in sample_images:
        img = X_train[idx]
        r_mean, g_mean, b_mean = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
        channel_means[class_name].append([r_mean, g_mean, b_mean])

df_channel_means = pd.DataFrame({
    "Class": np.repeat(class_labels, 10),
    "Red Mean": np.concatenate([np.array(channel_means[c])[:, 0] for c in class_labels]),
    "Green Mean": np.concatenate([np.array(channel_means[c])[:, 1] for c in class_labels]),
    "Blue Mean": np.concatenate([np.array(channel_means[c])[:, 2] for c in class_labels]),
})

plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Red Mean", data=df_channel_means, palette="Reds")
plt.title("Red Channel Mean Distribution Across Classes")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Green Mean", data=df_channel_means, palette="Greens")
plt.title("Green Channel Mean Distribution Across Classes")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Blue Mean", data=df_channel_means, palette="Blues")
plt.title("Blue Channel Mean Distribution Across Classes")
plt.show()

print("Exploratory Data Analysis (EDA) completed successfully.")
