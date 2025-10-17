# =============================================
# 6. CNN + GCN Model
# 6.1 Load Pretrained CNN (ResNet50) and Feature Extraction
# =============================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
