# =============================================
# 4.3 Load the Model from Keras Format
# =============================================

import os
import tensorflow as tf

# Define the model path (update if needed)
model_path = os.path.join("outputs", "optimized_hybrid_cnn_rnn_model.keras")

# Load the model
hybrid_model = tf.keras.models.load_model(model_path)

print("Model successfully loaded from `.keras` format at:", model_path)
