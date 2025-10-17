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

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model setup
gcn_model = GCN(data.num_node_features, 128, 3).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Train/validation split
num_nodes = data.num_nodes
train_idx, val_idx = train_test_split(torch.arange(num_nodes), test_size=0.2, stratify=data.y.cpu())
train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
train_mask[train_idx] = True
val_mask[val_idx] = True

# Trackers
training_loss, validation_loss = [], []
training_accuracy, validation_accuracy = [], []
best_val_acc = 0
best_model_state, best_predictions, best_out, best_y_true = None, None, None, None

# Training loop
for epoch in range(1, 201):
    gcn_model.train()
    optimizer.zero_grad()
    out = gcn_model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    # Loss & Accuracy
    training_loss.append(loss.item())
    val_loss = criterion(out[val_mask], data.y[val_mask]).item()
    validation_loss.append(val_loss)

    train_pred = out[train_mask].argmax(dim=1)
    val_pred = out[val_mask].argmax(dim=1)
    train_acc = train_pred.eq(data.y[train_mask]).float().mean().item()
    val_acc = val_pred.eq(data.y[val_mask]).float().mean().item()
    training_accuracy.append(train_acc)
    validation_accuracy.append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = gcn_model.state_dict()
        best_predictions = out.argmax(dim=1).cpu()
        best_out = out.detach().cpu()
        best_y_true = data.y.cpu()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Load best model
gcn_model.load_state_dict(best_model_state)

# Classification Report
y_true = best_y_true.numpy()
y_pred = best_predictions.numpy()
class_labels = ["Non-Tumor", "Non-Viable-Tumor", "Viable"]
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))

# ROC Curve
y_true_onehot = np.eye(len(class_labels))[y_true]
y_probs = torch.exp(best_out).numpy()

plt.figure(figsize=(7, 6))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# Loss Plot
plt.figure(figsize=(7, 4))
plt.plot(training_loss, label="Train Loss", color="red")
plt.plot(validation_loss, label="Val Loss", linestyle="dashed", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy Plot
plt.figure(figsize=(7, 4))
plt.plot(training_accuracy, label="Train Acc", color="green")
plt.plot(validation_accuracy, label="Val Acc", linestyle="dashed", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
