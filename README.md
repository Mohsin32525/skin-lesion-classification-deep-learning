# Skin Lesion Classification Using Advanced Deep Learning

## Overview

Skin cancer is one of the most common and potentially life-threatening diseases worldwide. Early detection plays a crucial role in improving patient survival rates.

This project presents an advanced deep learning pipeline for automatic classification of dermatoscopic images using the HAM10000 dataset. The work focuses on addressing class imbalance, improving model design, and applying rigorous evaluation techniques.

---

## Problem Statement

The objective is to classify skin lesion images into seven categories using deep learning. This is a multi-class classification problem with significant class imbalance, making evaluation and model design challenging.

---

## Dataset: HAM10000

- Total Images: 10,015  
- Number of Classes: 7  
- Domain: Medical Imaging  

### Classes:

- Melanoma (mel)  
- Melanocytic nevi (nv)  
- Basal cell carcinoma (bcc)  
- Actinic keratoses (akiec)  
- Benign keratosis (bkl)  
- Dermatofibroma (df)  
- Vascular lesions (vasc)  

### Key Challenge:

- Severe class imbalance (nv dominates dataset)

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## Methodology

### Baseline Limitation

A basic CNN model initially achieved ~68–69% accuracy but failed to detect minority classes, leading to majority class bias.

---

### Improved Approach

To address these issues, the pipeline was redesigned:

- Applied **class imbalance handling**:
  - Class weighting
  - Data augmentation
  - Strategic undersampling

- Used **Transfer Learning (MobileNetV2)** for better feature extraction

- Implemented **two-phase training**:
  1. Feature extraction (frozen base model)
  2. Fine-tuning (unfrozen model with low learning rate)

- Applied **EarlyStopping and learning rate control**

---

## Model Architecture

- MobileNetV2 (pre-trained on ImageNet)
- Global Average Pooling
- Dense layer (ReLU)
- Dropout (0.5)
- Softmax output layer (7 classes)

---

## Results

| Metric            | Value |
|------------------|------|
| Accuracy         | ~57% |
| Macro Recall     | ~0.52 |
| Macro F1-score   | ~0.43 |

---

## Key Results

### Confusion Matrix
![Confusion Matrix](figures/confusion_matrix.png)

### Classification Metrics
![Metrics](figures/classification_metrics.png)

---

## Key Observations

- Model successfully predicts **all classes** (no collapse)
- High recall achieved for minority classes:
  - vasc: 0.81  
  - df: 0.59  
- Performance is now **balanced across classes**

---

## Key Insights

- Accuracy alone is misleading for imbalanced datasets  
- Macro-F1 and recall provide a more realistic evaluation  
- Transfer learning significantly improves performance  
- Proper handling of imbalance is critical in medical applications  

---

## Project Structure

skin-lesion-classification/
│
├── notebook.ipynb
├── report.pdf
├── figures/
│ ├── class_distribution.png
│ ├── confusion_matrix.png
│ └── classification_metrics.png


---

## Future Work

- Apply focal loss for better imbalance handling  
- Explore EfficientNet for improved feature representation  
- Optimize classification thresholds for critical classes  
- Perform cross-validation for more robust evaluation  

---

## Author

Mohsin
