# Results

This section presents all the experimental results, dataset statistics, model evaluations, and performance metrics obtained during the **Soft Tissue Sarcoma Detection** project. The results include dataset distribution, feature extraction outcomes, and model performance summaries for both **CNN** and **CNN + GCN hybrid** architectures.

---

## 1. Dataset Distribution

### **Training Set**

| Class               | Number of Images |
|----------------------|------------------|
| Non-Tumor            | 340              |
| Non-Viable-Tumor     | 144              |
| Viable               | 156              |

### **Validation Set**

| Class               | Number of Images |
|----------------------|------------------|
| Non-Tumor            | 88               |
| Non-Viable-Tumor     | 66               |
| Viable               | 77               |

### **Test Set**

| Class               | Number of Images |
|----------------------|------------------|
| Non-Tumor            | 91               |
| Non-Viable-Tumor     | 42               |
| Viable               | 44               |

---

## 2. Dataset Overview

| Dataset     | Image Shape     | Total Images | Classes |
|--------------|------------------|---------------|----------|
| Train        | (224, 224, 3)    | 1020          | 3        |
| Validation   | (224, 224, 3)    | 264           | 3        |
| Test         | (224, 224, 3)    | 273           | 3        |

**Mean Pixel Intensity:** 0.7616  
**Standard Deviation of Pixel Intensities:** 0.2180  

---

## 3. Label Distribution (After Balancing)

| Class               | Number of Images |
|----------------------|------------------|
| Non-Tumor            | 340              |
| Non-Viable-Tumor     | 340              |
| Viable               | 340              |

---

## 4. CNN Model Results

**Test Accuracy:** `0.9451`

### **Classification Report**

| Class              | Precision | Recall | F1-Score | Support |
|---------------------|------------|--------|-----------|----------|
| Non-Tumor           | 0.86       | 0.98   | 0.92      | 91       |
| Non-Viable-Tumor    | 0.97       | 0.91   | 0.94      | 91       |
| Viable              | 0.99       | 0.91   | 0.95      | 91       |
| **Accuracy**        |            |        | **0.93**  | **273**  |
| **Macro Avg**       | 0.94       | 0.93   | 0.93      | 273      |
| **Weighted Avg**    | 0.94       | 0.93   | 0.93      | 273      |

---

## 5. CNN + GCN Model Results

**Test Accuracy:** `0.8059`

**Graph Statistics:**
- **Nodes:** 1020  
- **Edges:** 215,762 (before PyG conversion)  
- **Edges (PyG format):** 431,524  
- **Classes:** [0, 1, 2]  
- **Label Distribution:** [340, 340, 340]  

### **Model Performance**

| Metric                  | Value    |
|--------------------------|----------|
| Final Training Accuracy  | 0.8603   |
| Final Validation Accuracy| 0.8676   |

---

### **Classification Report**

| Metric          | Precision | Recall | F1-Score | Support |
|------------------|------------|--------|-----------|----------|
| Non-Tumor        | 0.84       | 0.87   | 0.85      | -        |
| Non-Viable-Tumor | 0.85       | 0.84   | 0.84      | -        |
| Viable           | 0.86       | 0.83   | 0.84      | -        |
| **Accuracy**     |            |        | **0.85**  | **1020** |
| **Macro Avg**    | 0.85       | 0.85   | 0.85      | 1020     |
| **Weighted Avg** | 0.85       | 0.85   | 0.85      | 1020     |

---
