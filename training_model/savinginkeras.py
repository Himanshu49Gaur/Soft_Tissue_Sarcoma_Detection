# =============================================
# 4.2 Save the Model in Keras Format
# =============================================

import os
import tensorflow as tf

# Define the output directory and file name
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
saved_model_path = os.path.join(output_dir, "optimized_hybrid_cnn_rnn_model.keras")

# Save the model in `.keras` format
hybrid_model.save(saved_model_path)

print("Model successfully saved in `.keras` format at:", saved_model_path)
