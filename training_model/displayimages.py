"""
# 1.3 Display Sample Images
## Objective:
Visualize sample images from each class in the training dataset to ensure data quality.

## Details:
- Displays a few images per class for visual inspection.
- Helps confirm that the dataset is correctly organized and readable.
"""

import os
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

def count_images_per_class(base_path):
    class_counts = defaultdict(int)
    
    if not os.path.exists(base_path):
        print(f"Warning: Path not found: {base_path}")
        return {}

    for class_folder in os.listdir(base_path):
        class_path = os.path.join(base_path, class_folder)
        if os.path.isdir(class_path):
            num_files = sum(
                1 for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            )
            class_counts[class_folder] = num_files

    return dict(class_counts)

def show_images(base_path, class_name, num_images=5):
    class_path = os.path.join(base_path, class_name)
    
    if not os.path.exists(class_path):
        print(f"Class directory not found: {class_path}")
        return
    
    all_files = [
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f)) and not f.startswith('.')
    ]
    
    if not all_files:
        print(f"No images found in {class_path}")
        return
    
    images = all_files[:num_images]

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    
    if len(images) == 1:
        axes = [axes]
    
    for ax, img_name in zip(axes, images):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            ax.axis('off')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(class_name)

    plt.tight_layout()
    plt.show()

dataset_path = "path_to_your_dataset_directory"
train_counts = count_images_per_class(os.path.join(dataset_path, "train"))

train_path = os.path.join(dataset_path, "train")
for class_name in train_counts.keys():
    show_images(train_path, class_name)
