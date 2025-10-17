# =============================================
# 6.4–6.5 Define and Train Graph Convolutional Network (GCN)
# =============================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

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

# Model parameters
hidden_dim = 128
num_classes = 3
gcn_model = GCN(in_features=data.x.shape[1], hidden_dim=hidden_dim, out_features=num_classes)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gcn_model = gcn_model.to(device)
data = data.to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Train/validation split
num_nodes = data.num_nodes
train_idx, val_idx = train_test_split(torch.arange(num_nodes), test_size=0.2, stratify=data.y.cpu())
train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
train_mask[train_idx] = True
val_mask[val_idx] = True

# Training function
def train():
    gcn_model.train()
    optimizer.zero_grad()
    out = gcn_model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate(mask):
    gcn_model.eval()
    with torch.no_grad():
        out = gcn_model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        correct = pred.eq(data.y[mask]).sum().item()
        accuracy = correct / mask.sum().item()
    return accuracy

# Training loop
for epoch in range(1, 201):
    loss = train()
    train_acc = evaluate(train_mask)
    val_acc = evaluate(val_mask)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Final accuracy report
final_train_acc = evaluate(train_mask)
final_val_acc = evaluate(val_mask)
print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
