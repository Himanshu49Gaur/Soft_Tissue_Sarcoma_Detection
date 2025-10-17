# =============================================
# 4.4 Model Evaluation
# =============================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)

preprocessed_dir = "/kaggle/input/preprocessed-osteosarcoma/preprocessed_osteosarcoma"
if not os.path.exists(preprocessed_dir):
    raise FileNotFoundError(f"Preprocessed directory not found at {preprocessed_dir}")

X_test = np.load(os.path.join(preprocessed_dir, "X_test_tensor.npy"))
y_test = np.load(os.path.join(preprocessed_dir, "y_test_tensor.npy"))

y_pred_probs = hybrid_model.predict([X_test, X_test])
if np.isnan(y_pred_probs).any():
    y_pred_probs = np.nan_to_num(y_pred_probs)

num_classes = len(np.unique(y_test))
y_test_onehot = np.eye(num_classes)[y_test]
y_pred = np.argmax(y_pred_probs, axis=1)
class_labels = ["Non-Tumor", "Non-Viable-Tumor", "Viable"]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(7, 6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_labels[i]} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 6))
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_test_onehot[:, i], y_pred_probs[:, i])
    ap_score = average_precision_score(y_test_onehot[:, i], y_pred_probs[:, i])
    plt.plot(recall, precision, lw=2, label=f'{class_labels[i]} (AP = {ap_score:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-Class Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1-score: {macro_f1:.4f}")

history_path = "/kaggle/working/training_history.npy"

def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

if os.path.exists(history_path):
    history = np.load(history_path, allow_pickle=True).item()

    plt.figure(figsize=(6, 4))
    plt.plot(smooth_curve(history["accuracy"]), label="Train Accuracy", marker="o")
    plt.plot(smooth_curve(history["val_accuracy"]), label="Validation Accuracy", linestyle="dashed", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(smooth_curve(history["loss"]), label="Train Loss", marker="o", color="red")
    plt.plot(smooth_curve(history["val_loss"]), label="Validation Loss", linestyle="dashed", marker="s", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()
