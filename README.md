#  Skin Cancer Classification using Deep Learning (CNN)

##  Overview

Skin cancer is one of the most common and potentially life-threatening diseases worldwide. Early detection plays a crucial role in improving survival rates. This project presents a deep learning-based approach for automatic classification of skin lesion images using a Convolutional Neural Network (CNN).

The model is trained on the HAM10000 dataset and demonstrates how deep learning can assist in medical image diagnosis.

---

##  Problem Statement

The goal of this project is to classify dermatoscopic images of skin lesions into multiple categories using deep learning techniques. This is a multi-class classification problem with significant class imbalance.

---

##  Dataset: Skin Cancer MNIST – HAM10000

* **Domain:** Medical Imaging
* **Total Images:** 10,015
* **Number of Classes:** 7
* **Task:** Multi-class classification

###  Classes Include:

* Melanoma
* Melanocytic nevi
* Basal cell carcinoma
* Actinic keratoses
* Benign keratosis
* Dermatofibroma
* Vascular lesions

###  Key Challenges:

* Severe **class imbalance** (dominant class: melanocytic nevi)
* Requires **image preprocessing and augmentation**
* Real-world medical dataset complexity

### 🔗 Dataset Links:

*  Research Paper: https://arxiv.org/abs/1803.10417
*  Kaggle Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

##  Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn

---

##  Project Workflow

1. Data Loading and Preprocessing
2. Image Path Generation
3. Data Cleaning (removal of missing images)
4. Exploratory Data Analysis
5. Label Encoding
6. Train-Test Split (Stratified)
7. Data Augmentation
8. CNN Model Development
9. Model Training
10. Model Evaluation

---

##  Model Architecture

A Convolutional Neural Network (CNN) is implemented with:

* Convolutional layers for feature extraction
* MaxPooling layers for spatial reduction
* Fully connected layers for classification
* Dropout layer to prevent overfitting
* Softmax output layer for multi-class classification

---

## 📈 Results

| Metric              | Value    |
| ------------------- | -------- |
| Training Accuracy   | ~68%     |
| Validation Accuracy | ~68%     |
| Test Accuracy       | **~69%** |

 **Class Distribution**
![Class Distribution](figures/class_distribution.png)

 **Accuracy Curve**
![Accuracy Graph](figures/accuracy_plot.png)

 **Confusion Matrix**
![Confusion Matrix](figures/confusion_matrix.png)

---

##  Evaluation

* The model performs well on majority classes
* Poor performance observed on minority classes
* Indicates bias due to class imbalance

---

##  Challenges

### Class Imbalance

The dataset is highly imbalanced, causing:

* Bias toward dominant classes
* Low recall and precision for minority classes
* Misleading overall accuracy

---

##  Future Improvements

* Apply **transfer learning** (ResNet, EfficientNet)
* Use **class weighting / oversampling**
* Train on full dataset
* Hyperparameter tuning
* Advanced augmentation techniques

---

##  Model Saving

```python
model.save("skin_cancer_model.h5")
```

---

##  Project Structure

```
skin-cancer-classification-cnn/
│
├── dataset/   (not uploaded due to large size)
├── figures/
│   ├── class_distribution.png
│   ├── accuracy_plot.png
│   └── confusion_matrix.png
│
├── skin_cancer_cnn.ipynb
└── README.md
```

---

##  Key Insights

* Deep learning is effective for medical image classification
* Dataset quality and balance significantly impact performance
* Evaluation metrics beyond accuracy are essential

---

## 🏁 Conclusion

This project demonstrates the application of CNNs for skin cancer classification using the HAM10000 dataset. The model achieved approximately 69% accuracy, showing promising results.

However, class imbalance remains a major limitation and highlights the need for more advanced techniques in real-world applications.

---

##  Author

Mohsin

---
Note: Dataset is not included due to size limitations. Please download from Kaggle link above.
