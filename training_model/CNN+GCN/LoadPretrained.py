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

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
resnet50.to(device)
resnet50.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(model, data, batch_size=16):
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
    features_list = []
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = preprocess(batch) if preprocess else batch
            batch = batch.to(device)
            output = model(batch)
            output = output.view(output.size(0), -1).cpu().numpy()
            features_list.append(output)

    return np.vstack(features_list)

data_path = "/kaggle/input/preprocessed-osteosarcoma/preprocessed_osteosarcoma/"
X_train = np.load(data_path + "X_train_tensor.npy")
X_val = np.load(data_path + "X_validate_tensor.npy")
X_test = np.load(data_path + "X_test_tensor.npy")

X_train_features = extract_features(resnet50, X_train, batch_size=32)
X_val_features = extract_features(resnet50, X_val, batch_size=32)
X_test_features = extract_features(resnet50, X_test, batch_size=32)

X_train_features = normalize(X_train_features, axis=1)
X_val_features = normalize(X_val_features, axis=1)
X_test_features = normalize(X_test_features, axis=1)

print("CNN Features Extracted:")
print("Train:", X_train_features.shape)
print("Val:", X_val_features.shape)
print("Test:", X_test_features.shape)
