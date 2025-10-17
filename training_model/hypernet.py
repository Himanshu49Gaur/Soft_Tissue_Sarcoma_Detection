# =============================================
# 4. HyPerNet: Hybrid ResNet50-EfficientNetB0 with LSTM-GRU Fusion
# =============================================
# 4.1 Model Training

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Concatenate,
    Bidirectional, LSTM, GRU, GlobalAveragePooling2D, Reshape
)
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Enable Mixed Precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# =============================================
# Load Preprocessed Dataset
# =============================================

preprocessed_dir = "path_to_preprocessed_dataset"

X_train = np.load(os.path.join(preprocessed_dir, "X_train_tensor.npy"))
y_train = np.load(os.path.join(preprocessed_dir, "y_train_tensor.npy"))
X_val = np.load(os.path.join(preprocessed_dir, "X_validate_tensor.npy"))
y_val = np.load(os.path.join(preprocessed_dir, "y_validate_tensor.npy"))
X_test = np.load(os.path.join(preprocessed_dir, "X_test_tensor.npy"))
y_test = np.load(os.path.join(preprocessed_dir, "y_test_tensor.npy"))

# =============================================
# Model Definition
# =============================================

input_shape = (224, 224, 3)
num_classes = 3

resnet_input = Input(shape=input_shape, name="resnet_input")
effnet_input = Input(shape=input_shape, name="effnet_input")

resnet = ResNet50(weights="imagenet", include_top=False, input_tensor=resnet_input)
efficientnet = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=effnet_input)

for layer in resnet.layers:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

for layer in efficientnet.layers:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

resnet_features = GlobalAveragePooling2D()(resnet.output)
efficientnet_features = GlobalAveragePooling2D()(efficientnet.output)

resnet_reshape = Reshape((1, 2048))(resnet_features)
effnet_reshape = Reshape((1, 1280))(efficientnet_features)

resnet_lstm = Bidirectional(LSTM(128, return_sequences=False))(resnet_reshape)
effnet_gru = GRU(128, return_sequences=False)(effnet_reshape)

merged = Concatenate()([resnet_lstm, effnet_gru])

x = Dense(256, activation="relu")(merged)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

output = Dense(num_classes, activation="softmax", dtype="float32")(x)

hybrid_model = Model(inputs=[resnet_input, effnet_input], outputs=output)

hybrid_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"]
)

# =============================================
# Data Augmentation
# =============================================

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

def augment_images(X_data):
    return np.array([datagen.random_transform(img) for img in X_data])

X_train_resnet = augment_images(X_train)
X_train_effnet = augment_images(X_train)
X_val_resnet = augment_images(X_val)
X_val_effnet = augment_images(X_val)
X_test_resnet = augment_images(X_test)
X_test_effnet = augment_images(X_test)

# =============================================
# Class Weights & Training
# =============================================

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

history = hybrid_model.fit(
    [X_train_resnet, X_train_effnet], y_train,
    validation_data=([X_val_resnet, X_val_effnet], y_val),
    epochs=30,
    batch_size=16,
    class_weight=class_weights_dict,
    verbose=1
)

# =============================================
# Model Evaluation & Saving
# =============================================

os.makedirs("outputs", exist_ok=True)
np.save("outputs/training_history.npy", history.history)

test_loss, test_accuracy = hybrid_model.evaluate([X_test_resnet, X_test_effnet], y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")

hybrid_model.save("outputs/optimized_hybrid_cnn_rnn_model.h5")
