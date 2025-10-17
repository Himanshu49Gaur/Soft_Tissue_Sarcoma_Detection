# =============================================
# 5. CNN Model
# 5.1 Model Training
# =============================================

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

data_path = "/kaggle/input/preprocessed-osteosarcoma/preprocessed_osteosarcoma/"

X_train = np.load(data_path + "X_train_tensor.npy")
y_train = np.load(data_path + "y_train_tensor.npy")
X_val = np.load(data_path + "X_validate_tensor.npy")
y_val = np.load(data_path + "y_validate_tensor.npy")
X_test = np.load(data_path + "X_test_tensor.npy")
y_test = np.load(data_path + "y_test_tensor.npy")
class_weights = np.load(data_path + "class_weights.npy", allow_pickle=True).item()

def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = X_train.shape[1:]
cnn_model = build_cnn_model(input_shape)

history = cnn_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3)
    ]
)

test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
