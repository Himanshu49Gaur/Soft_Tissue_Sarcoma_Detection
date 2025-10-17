# =============================================
# 6.3 Convert NetworkX Graph to PyTorch Geometric Format
# =============================================

import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx

# Convert adjacency matrix to PyTorch edge_index
adj_matrix = nx.to_numpy_array(graph)
edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)

# Extract node features and standardize
node_features = np.array([graph.nodes[n]['feature'] for n in graph.nodes])
x = torch.tensor(node_features, dtype=torch.float)
x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

# Extract node labels
node_labels = np.array([graph.nodes[n]['label'] for n in graph.nodes])
y = torch.tensor(node_labels, dtype=torch.long)

# Create PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=y)

print(f"Graph converted to PyG format. Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Label distribution: {torch.bincount(data.y)}")
