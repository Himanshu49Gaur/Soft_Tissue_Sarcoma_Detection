# Exploratory Data Analysis (EDA)

## 1. Overview

Exploratory Data Analysis (EDA) is a crucial phase of this project aimed at understanding the structure, distribution, and intrinsic properties of the **Soft Tissue Sarcoma** dataset before applying any machine learning or deep learning techniques.  
EDA helps identify patterns, detect anomalies, and ensure the dataset is balanced, clean, and suitable for model training.

The EDA conducted in this section provided insights into **data quality, class distribution, feature variability, and normalization requirements**, all of which guided the model architecture design.

---

## 2. Objectives of EDA

1. **Understand Dataset Composition** – Determine the number of images, their dimensions, and class distribution across training, validation, and test sets.  
2. **Assess Class Balance** – Identify if the dataset is imbalanced and perform augmentation or sampling where necessary.  
3. **Visualize Data Distribution** – Generate visual summaries to understand pixel intensity, color channels, and feature variation.  
4. **Detect Anomalies or Noise** – Identify corrupted or misclassified samples through visualization and statistical metrics.  
5. **Guide Preprocessing Decisions** – Provide a data-driven foundation for normalization, augmentation, and feature extraction strategies.

---

## 3. Techniques Used

The following EDA techniques and methods were employed in this project:

| Technique | Description | Implementation Summary |
|------------|-------------|-------------------------|
| **Dataset Summary Statistics** | Computes total samples, mean, standard deviation, and dimensionality across all sets. | Used Python’s `os` and `PIL` to traverse image directories and calculate statistics. |
| **Class Distribution Analysis** | Evaluates image count per label to identify imbalance. | Implemented with `pandas` and `matplotlib` for count plots. |
| **Image Visualization** | Displays random image samples from each class to verify data quality. | Utilized `matplotlib.pyplot.imshow()` for visual inspection. |
| **Pixel Intensity Distribution** | Examines mean and standard deviation of pixel intensities to check brightness variation. | Used NumPy for pixel-level analysis. |
| **Feature Correlation Study** | Identifies correlations or redundancies between extracted features. | Conducted with `seaborn.heatmap()` and `numpy.corrcoef()`. |
| **Augmentation Impact Review** | Assesses the effectiveness of data augmentation in balancing classes. | Compared pre- and post-augmentation histograms. |

---

## 4. Implementation Summary

All the above techniques are implemented in the EDA code file — `eda_analysis.py`.  
The code follows a structured approach:

1. **Data Loading:** Reads and organizes image data from `train`, `validation`, and `test` folders.  
2. **Statistical Computation:** Calculates descriptive statistics for the dataset.  
3. **Visualization:** Generates graphs and plots for class balance, pixel intensity, and sample previews.  
4. **Data Integrity Check:** Identifies missing, corrupted, or incorrectly labeled files.  
5. **Feature-Level Analysis:** Reviews extracted features and their relationships across samples.

---

## 5. Results and Observations

| Metric | Observation |
|--------|--------------|
| **Image Resolution** | All images standardized to (224 × 224 × 3). |
| **Number of Classes** | 3 classes – Non-Tumor, Non-Viable Tumor, Viable. |
| **Class Balance (Before Balancing)** | Non-Tumor: 340, Non-Viable-Tumor: 144, Viable: 156. |
| **Class Balance (After Balancing)** | All classes balanced to 340 samples each. |
| **Mean Pixel Intensity** | 0.7616 |
| **Standard Deviation** | 0.2180 |
| **Augmentation Effectiveness** | Successfully increased dataset diversity and balance without loss of image quality. |

**Key Insight:**  
The EDA revealed slight class imbalance and high feature variance, which justified implementing augmentation and normalization. These steps improved model convergence and reduced bias in later training stages.

---

## 6. Files Included

| File Name | Description |
|------------|-------------|
| `eda_analysis.py` | Python File containing full exploratory data analysis workflow. |
| `README.md` | Documentation explaining the purpose, methodology, and findings of the EDA process. |

---

## 7. Conclusion

The Exploratory Data Analysis phase provided a comprehensive understanding of the Soft Tissue Sarcoma dataset, ensuring the integrity, quality, and readiness of the data for deep learning model development.  
The insights gained here directly influenced preprocessing, augmentation, and model selection, leading to improved accuracy and stability in the final detection framework.
