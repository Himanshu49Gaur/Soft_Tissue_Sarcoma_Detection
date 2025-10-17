# =============================================
# 6.2 Create Graph from CNN Features
# =============================================

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data_path = "/kaggle/input/preprocessed-osteosarcoma/preprocessed_osteosarcoma/"
y_train = np.load(data_path + "y_train_tensor.npy")

features = X_train_features  # Extracted CNN features
num_images = features.shape[0]

if len(y_train) != num_images:
    raise ValueError(f"Mismatch: {len(y_train)} labels vs {num_images} features")

graph = nx.Graph()

for i in range(num_images):
    graph.add_node(i, feature=features[i], label=int(y_train[i]))

print(f"{num_images} nodes added with features and labels")

features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

print("Computing cosine similarity matrix...")
similarity_matrix = cosine_similarity(features_norm)

threshold = 0.7
edge_count = 0

for i in range(num_images):
    for j in range(i + 1, num_images):
        if similarity_matrix[i, j] > threshold:
            graph.add_edge(i, j, weight=similarity_matrix[i, j])
            edge_count += 1

print(f"Graph created with {graph.number_of_nodes()} nodes and {edge_count} edges")

node_labels = np.array([graph.nodes[n]['label'] for n in graph.nodes])
unique_labels, label_counts = np.unique(node_labels, return_counts=True)
print(f"Unique Labels: {unique_labels.tolist()} - Counts: {label_counts.tolist()}")
