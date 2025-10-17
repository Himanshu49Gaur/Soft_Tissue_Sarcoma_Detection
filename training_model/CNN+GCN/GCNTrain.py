# =============================================
# 6.4–6.5 Define and Train Graph Convolutional Network (GCN)
# =============================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
