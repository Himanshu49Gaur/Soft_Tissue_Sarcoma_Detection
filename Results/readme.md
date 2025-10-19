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
