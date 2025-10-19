# Soft Tissue Sarcoma Detection using Graph Convolutional Networks (GCN)

## 1. Introduction
Soft Tissue Sarcoma (STS) refers to a diverse group of malignant tumors arising from connective tissues such as muscles, fat, nerves, and blood vessels. Accurate diagnosis and classification of STS are critical for effective treatment planning, yet conventional diagnostic approaches are largely manual and subjective.

This project presents a **Graph Convolutional Network (GCN)**-based framework for automated detection and classification of soft tissue sarcoma. By representing feature interactions as graph structures, the system captures spatial and relational dependencies that traditional deep learning models often overlook. The final model classifies tissue states into three categories:

- Non-Tumor  
- Non-Viable Tumor  
- Viable Tumor  

The proposed pipeline combines data preprocessing, hybrid feature extraction, graph construction, and graph-based deep learning to improve the accuracy and interpretability of STS classification.

---

## 2. Objectives
- Construct a robust end-to-end pipeline for soft tissue sarcoma detection using graph-based learning.  
- Employ Graph Convolutional Networks (GCNs) for capturing non-linear dependencies among tissue features.  
- Analyze the performance of the proposed model using standard evaluation metrics and visualization tools.

---

## 3. Methodology

### 3.1 Data Preprocessing
- Data is cleaned, normalized, and formatted to ensure uniformity.  
- Feature vectors are extracted and represented as nodes in a graph.  
- An adjacency matrix or edge list defines relationships between data points based on feature similarity or distance metrics.

### 3.2 Model Loading
- Pretrained hybrid **CNN-RNN** models saved in `.keras` format are loaded using TensorFlow.  
- These models provide feature-level representations for subsequent graph learning.

### 3.3 Graph Construction
- Dataset converted into a **PyTorch Geometric** `Data` object consisting of:
  - Node features (x)  
  - Edge indices (edge_index)  
  - Target labels (y)  
- Data transferred to GPU for accelerated computation.

### 3.4 Model Definition (GCN)
- Implemented using `torch_geometric.nn.GCNConv`.  
- Architecture: two convolutional layers with **ReLU** and **log-softmax** activation.  
- Hidden dimension: 128  
- Output classes: 3

### 3.5 Model Training
- Dataset split into training and validation subsets using stratified sampling.  
- Trained for **200 epochs** using **Adam optimizer** (learning rate: 0.001, weight decay: 5e-4).  
- **Cross-entropy loss** used as the objective function.  
- Training and validation metrics logged per epoch.

### 3.6 Evaluation and Analysis
- Best-performing model (based on validation accuracy) saved.  
- Evaluation metrics include:
  - Classification report (Precision, Recall, F1-score)  
  - Confusion matrix  
  - ROC and AUC scores  
- Visualizations:
  - ROC curves  
  - Loss and accuracy curves  
  - Confusion matrix heatmap  

---

## 4. Technologies Used

| Category | Libraries / Frameworks |
|-----------|------------------------|
| Programming Language | Python 3.x |
| Deep Learning | PyTorch, TensorFlow |
| Graph Learning | PyTorch Geometric |
| Data Manipulation | NumPy, pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| GPU Acceleration | CUDA |

---

## 5. Key Features
- End-to-end automated soft tissue sarcoma detection pipeline.  
- Integration of hybrid CNN-RNN features with graph-based learning.  
- GPU-enabled model training for enhanced performance.  
- Modular code structure for scalability and experimentation.  
- Comprehensive visualization and interpretability components.

---

## 6. Results
The model achieved high validation accuracy with clear class separation across all three tissue types.

**Performance Metrics (Indicative):**

| Metric | Value |
|---------|--------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~93% |
| AUC (Average) | >0.90 |

**Visualization outputs include:**
- Training and validation loss/accuracy curves  
- ROC curves for multi-class classification  
- Confusion matrix for class-wise performance

---

## 7. File Description

| File | Description |
|------|--------------|
| `Soft_Tissue_Sarcoma_Detection.ipynb` | Jupyter notebook containing full pipeline & analysis |
| `Soft_Tissue_Sarcoma_Detection.py` | Executable Python version of the notebook |
| `README.md` | Documentation file |
| `Separate code files in python` | All the files with each cell excecuted in different python files avaliable in the model training section |
| `Documents Section`| Contains all the required docuementations of the project |
| `Result Section` | Contains all the results with Images, Results, Graphs etc. | 

---

## 8. How to Run

### Install Dependencies
pip install torch torchvision torchaudio torch-geometric tensorflow scikit-learn matplotlib seaborn numpy pandas

### Ensure CUDA Availability
import torch
print("CUDA Available:", torch.cuda.is_available())

### Execute the Script
python Soft_Tissue_Sarcoma_Detection.py

### Optional
To explore analysis interactively:
jupyter notebook Soft_Tissue_Sarcoma_Detection.ipynb

undefined

---

## 9. Discussion and Future Work
The GCN-based approach successfully models relational dependencies among features, improving classification performance in complex biomedical datasets.

**Future Enhancements:**
- Expand dataset for better generalization.  
- Implement **Graph Attention Networks (GAT)** for adaptive weighting.  
- Incorporate **explainable AI** techniques for enhanced interpretability.  
- Deploy trained model via **web/Streamlit** interface for clinical accessibility.

---


## **About the Author** 
| Matrix | Descrpition |
|--------|----------------|
| `Name` | Himanshu Gaur |
| `Email` | himanshugaur1810@gmail.com |
| `LinkedIn` | https://www.linkedin.com/in/himanshu-gaur-305006282/ |
| `GitHub` | Himanshu49Gaur |

---

## 10. References
- Kipf, T.N., & Welling, M. (2016). *Semi-Supervised Classification with Graph Convolutional Networks.*  
- Hamilton, W. et al. (2017). *Inductive Representation Learning on Large Graphs.*  
- PyTorch Geometric Documentation – [https://pytorch-geometric.readthedocs.io]()  
- TensorFlow Keras Documentation – [https://www.tensorflow.org/guide/keras]()
