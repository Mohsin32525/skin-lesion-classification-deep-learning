
#  Skin Cancer Classification Using Deep Learning

##  Project Overview

This project focuses on the classification of skin lesion images using a Convolutional Neural Network (CNN). The model is trained on the HAM10000 dataset to automatically identify different types of skin lesions.

Deep learning techniques are applied to analyze medical images and assist in early detection of skin cancer.

---

##  Objectives

* Develop a CNN model for multi-class skin lesion classification
* Perform data preprocessing and augmentation
* Evaluate model performance using multiple metrics
* Analyze the impact of class imbalance

---

##  Dataset

The dataset used is **HAM10000 (Human Against Machine with 10000 training images)**.

* Total images used: 3000 (sampled for faster training)
* Number of classes: 7
* Image format: JPG
* Metadata includes lesion type and image ID

⚠️ The dataset is imbalanced, meaning some classes have significantly more samples than others.

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn

---

##  Workflow

1. Data Loading and Preprocessing
2. Image Path Creation
3. Data Cleaning (removing missing images)
4. Data Visualization
5. Label Encoding
6. Train-Test Split
7. Data Augmentation
8. CNN Model Building
9. Model Training
10. Model Evaluation

---

##  Model Architecture

The CNN model consists of:

* Convolutional layers (feature extraction)
* MaxPooling layers (dimensionality reduction)
* Dense layers (classification)
* Dropout layer (prevent overfitting)
* Softmax output layer (multi-class classification)

---

##  Results

* Training Accuracy ≈ 68%
* Validation Accuracy ≈ 68%
* Test Accuracy ≈ **69%**

The model demonstrates stable learning and good generalization performance.

---

## 📊 Visualizations

### 🔹 Class Distribution

![Class Distribution](figures/class_distribution.png)

### 🔹 Accuracy Curve

![Accuracy Graph](figures/accuracy_plot.png)

### 🔹 Confusion Matrix

![Confusion Matrix](figures/confusion_matrix.png)

---

##  Challenges

### Class Imbalance

The dataset contains uneven distribution across classes, causing the model to perform better on dominant classes and poorly on minority classes.

This is reflected in the classification report, where some classes have low precision and recall.

---

## 🚀 Future Improvements

* Use transfer learning (ResNet, EfficientNet)
* Apply class weighting or oversampling
* Increase dataset size
* Hyperparameter tuning
* Advanced augmentation techniques

---

##  Model Saving

The trained model can be saved using:

```python
model.save("skin_cancer_model.h5")
```

---

## 📂 Project Structure

```
skin_cancer_project/
│
├── dataset/
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   └── HAM10000_metadata.csv
│
├── skin_cancer_cnn.ipynb
├── figures/
│   ├── class_distribution.png
│   ├── accuracy_plot.png
│   └── confusion_matrix.png
│
└── README.md
```

---

##  Conclusion

A CNN-based deep learning model was successfully developed for skin lesion classification. The model achieved approximately 69% accuracy, demonstrating the potential of deep learning in medical image analysis.

However, class imbalance remains a major challenge and needs to be addressed for improved performance.

---

##  Author

Mohsin

---

