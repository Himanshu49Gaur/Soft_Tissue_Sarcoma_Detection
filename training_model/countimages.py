"""
# 1.2 Count Images in Train, Test, and Validation Sets
## Objective:
Count the number of image files in each class subdirectory for every dataset split.

## Details:
- Iterates over 'train', 'validate', and 'test' directories.
- Displays the image count per class in each split.
- Helps verify balanced data distribution before training.
"""

import os
from collections import defaultdict

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

def print_dataset_distribution(dataset_path):
    splits = ['train', 'validate', 'test']
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        counts = count_images_per_class(split_path)
        print(f"\n{split.capitalize()} Set Distribution:")
        if counts:
            for cls, count in counts.items():
                print(f"  Class '{cls}': {count} images")
        else:
            print("  No data found or path missing.")

dataset_path = "path_to_your_dataset_directory"
print_dataset_distribution(dataset_path)
