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
