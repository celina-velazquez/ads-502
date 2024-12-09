# Breast Cancer Classification Project

## Project Overview

This project explores the application of machine learning techniques to classify breast cancer tumors as **malignant** or **benign**, leveraging the **Breast Cancer Wisconsin (Diagnostic) Dataset**. Early detection is critical for improving patient outcomes, and this work demonstrates how data-driven approaches like Support Vector Machines (SVM) and Artificial Neural Networks (ANN) can enhance diagnostic accuracy.

### Course

- **University**: University of San Diego
- **Program**: MS in Applied Data Science
- **Course**: ADS 502 - Applied Data Mining

### Dataset

- **Source**: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Details**: 
  - 569 observations with 32 columns, including 30 numeric features.
  - Features include tumor size, texture, symmetry, and fractal dimensions.
  - The target variable (`diagnosis`) is binary: `B` for benign and `M` for malignant.

## Authors

- **Celina Velazquez**
- **Kiara Paz**
- **Jordan Torres**

## Exploratory Data Analysis (EDA)

The dataset was carefully explored to understand its structure and prepare it for modeling:
- **Key Insights**:
  - Tumor size-related variables like `radius_worst` and `area_worst` are strong predictors of malignancy.
  - Texture-related variables exhibit balanced variability across diagnoses.
  - Correlation analysis revealed multicollinearity among size-related features, informing feature selection.
- **Visualizations**:
  - Correlation heatmaps identified highly correlated variables.
  - Boxplots and histograms highlighted class separability for significant features.

## Machine Learning Models

### Support Vector Machines (SVM)
- **Best Kernel**: Linear Kernel, achieving:
  - **Accuracy**: 96%
  - **Precision**: 97%
  - **Recall**: 95%
- **Preprocessing**:
  - Standardized all features using `StandardScaler`.
  - Addressed slight class imbalance with stratified cross-validation.
- **Other Kernels Tested**: Radial Basis Function (RBF) and Polynomial Kernels, which showed lower performance and signs of overfitting.

### Artificial Neural Networks (ANN)
- **Architecture**:
  - Input layer with 10 neurons.
  - One hidden layer with 10 neurons and ReLU activation.
  - Output layer with 1 neuron and sigmoid activation for binary classification.
- **Performance**:
  - **Accuracy**: 90%
  - **AUC**: 0.95
  - **Recall**: 76%
  - **Precision**: 92%
- **Optimization**: Used the Adam optimizer to mitigate issues like vanishing gradients.

## Performance Metrics

Evaluation of the models was based on:
- **Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Key Observations**:
  - SVM outperformed ANN in both accuracy and recall, making it the more reliable model for early diagnosis.
  - ANN demonstrated flexibility in capturing non-linear relationships, but recall performance needs improvement for high-stakes applications like cancer detection.

## Tools and Libraries Used

- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy` for data manipulation.
  - `matplotlib`, `seaborn` for visualization.
  - `scikit-learn` for machine learning models and evaluation.
  - `tensorflow`/`keras` for building and training the ANN.

## Requirements

Install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Instructions to Run

1. Clone this repository and navigate to the project directory.
2. Ensure the dataset (`breast-cancer.csv`) is in the root directory.
3. Run the Jupyter Notebook `breast_cancer_ml.ipynb` to view the analysis and results.
4. For a quick summary, review the included visualizations and performance metrics.

## Future Work

- Expanding the dataset to test generalizability across diverse populations.
- Exploring ensemble methods like Random Forests and Gradient Boosting.
- Hyperparameter tuning for both SVM and ANN to optimize performance further.
