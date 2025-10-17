# =============================================
# 6.7–6.9: CNN + GCN Evaluation, ROC, Loss/Accuracy Plots
# =============================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
