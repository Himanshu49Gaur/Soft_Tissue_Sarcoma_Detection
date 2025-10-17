# =============================================
# 5. CNN Model
# 5.3 Testing on Sample Data
# =============================================

import matplotlib.pyplot as plt
import random
import numpy as np

class_names = {0: "Non-Tumor", 1: "Non-Viable-Tumor", 2: "Viable"}

sample_indices = random.sample(range(len(X_test)), 5)
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, idx in enumerate(sample_indices):
    img = X_test[idx]
    actual_label = int(y_test[idx])
    predicted_prob = cnn_model.predict(np.expand_dims(img, axis=0))
    predicted_label = np.argmax(predicted_prob)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"Pred: {class_names[predicted_label]}\nActual: {class_names[actual_label]}")

plt.tight_layout()
plt.show()
