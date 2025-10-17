# ================================================
# 2. Data Preprocessing
# ================================================
# 2.1 Define Augmentation Pipeline
# 2.2 Define Preprocessing Function (resize, normalize, augment)
# 2.3 Compute Class Weights
# 2.4 Convert NumPy Arrays to TensorFlow Tensors
# ================================================

import os
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
import albumentations as A
from sklearn.utils.class_weight import compute_class_weight

# Set dataset and output paths (edit these as needed)
dataset_path = "path_to_your_dataset"  # Example: "E:/Datasets/Osteosarcoma"
preprocessed_dir = "path_to_save_preprocessed_data"  # Example: "./preprocessed_osteosarcoma"

# Verify dataset existence
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please verify the path.")

train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "validate")
test_dir = os.path.join(dataset_path, "test")

# Create directories for preprocessed data
for split in ["train", "validate", "test"]:
    os.makedirs(os.path.join(preprocessed_dir, split), exist_ok=True)

# Define augmentation pipeline
augmentations = A.Compose([
    A.Rotate(limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ZoomBlur(max_factor=1.1, p=0.3),
])

def augment_data(images, target_count):
    augmented_images = []
    if len(images) == 0:
        return np.array([])
    while len(augmented_images) + len(images) < target_count:
        img = images[np.random.randint(0, len(images))]
        augmented = augmentations(image=(img * 255).astype(np.uint8))
        aug_img = augmented["image"] / 255.0
        augmented_images.append(aug_img)
    return np.array(augmented_images, dtype=np.float32)

def preprocess_and_save_numpy_balanced(base_dir, output_dir, dataset_name):
    X, y = [], []
    class_counts = {}

    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_class_count = 0
    images_per_class = {}

    for class_name in classes:
        class_path = os.path.join(base_dir, class_name)
        images = []
        for img_name in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            images.append(img)
        images_per_class[class_name] = images
        max_class_count = max(max_class_count, len(images))

    for label, class_name in enumerate(classes):
        images = images_per_class[class_name]
        count = len(images)
        class_counts[class_name] = count

        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        X.extend(images)
        y.extend([label] * count)

        if count < max_class_count:
            extra_augmented = augment_data(images, max_class_count)
            X.extend(extra_augmented)
            y.extend([label] * len(extra_augmented))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    np.save(os.path.join(preprocessed_dir, f"X_{dataset_name}.npy"), X)
    np.save(os.path.join(preprocessed_dir, f"y_{dataset_name}.npy"), y)

    print(f"Saved balanced {dataset_name} dataset: {X.shape[0]} samples")

    return X, y

# Process all datasets
X_train, y_train = preprocess_and_save_numpy_balanced(train_dir, os.path.join(preprocessed_dir, "train"), "train")
X_val, y_val = preprocess_and_save_numpy_balanced(val_dir, os.path.join(preprocessed_dir, "validate"), "validate")
X_test, y_test = preprocess_and_save_numpy_balanced(test_dir, os.path.join(preprocessed_dir, "test"), "test")

# Compute and save class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print("Computed class weights:", class_weights_dict)
np.save(os.path.join(preprocessed_dir, "class_weights.npy"), class_weights_dict)

# Convert numpy arrays to TensorFlow tensors and save
def save_tf_tensors(X, y, name):
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    np.save(os.path.join(preprocessed_dir, f"X_{name}_tensor.npy"), X_tensor.numpy())
    np.save(os.path.join(preprocessed_dir, f"y_{name}_tensor.npy"), y_tensor.numpy())

save_tf_tensors(X_train, y_train, "train")
save_tf_tensors(X_val, y_val, "validate")
save_tf_tensors(X_test, y_test, "test")

# Zip preprocessed data for easy sharing
zip_path = os.path.join(preprocessed_dir, "preprocessed_osteosarcoma.zip")
shutil.make_archive(zip_path.replace(".zip", ""), 'zip', preprocessed_dir)

print(f"Preprocessed data zipped at {zip_path}")
