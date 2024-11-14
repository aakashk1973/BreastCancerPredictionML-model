# Breast Cancer Prediction Model (Clinical Data Only)

## Project Overview

This project aims to build a **breast cancer prediction model** using **clinical data** such as age, family history, tumor size, and other demographic and medical features. The goal is to predict whether a tumor is benign or malignant based on clinical features, allowing for early diagnosis and better decision-making in the healthcare process.

### Key Features:
- **Clinical Data**: Utilizes clinical features like age, family history, tumor size, and other patient-related factors.
- **Data Cleaning & Preprocessing**: Extensive preprocessing to clean the data and handle missing values, outliers, and categorical variables, which enhances model performance.
- **Machine Learning Models**: Various machine learning algorithms used to classify tumor cases into benign or malignant categories.
- **Model Evaluation**: Achieved better classification metrics through feature engineering and hyperparameter tuning.

## Problem Statement

Breast cancer is one of the leading causes of death among women worldwide. Accurate and early diagnosis is critical for better treatment outcomes. This model uses clinical data to predict whether a breast tumor is benign or malignant, which can help healthcare professionals in the early diagnosis process.

## Approach & Solution

### Data Preprocessing:
1. **Cleaning the Data**:
   - Handled missing data using **mean imputation**, **KNN imputation**, or **drop methods** based on the context.
   - Identified and treated outliers using techniques like **z-score normalization** or **IQR (Interquartile Range)** method.
   
2. **Feature Engineering**:
   - Encoded categorical variables using **One-Hot Encoding** or **Label Encoding**.
   - Scaled continuous variables using **Min-Max Scaling** or **Standardization** to normalize the data.

3. **Modeling**:
   - Various machine learning models were tested, including **Logistic Regression**, **Random Forest**, **Support Vector Machines (SVM)**, and **XGBoost**.
   - Hyperparameter tuning was performed using **GridSearchCV** to find the best parameters for each model.
   
4. **Model Evaluation**:
   - The performance of the models was evaluated using **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.

## Technologies Used

- **Machine Learning**: Scikit-learn, XGBoost, Random Forest, SVM
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Scikit-learn (Metrics and Cross-validation)
- **API Framework**: Flask

## Flask API Endpoint

This project provides a **Flask API** that can be used to predict whether a tumor is benign or malignant based on clinical features provided as input.

### Available Endpoints:

1. **POST /predict**: Predict breast cancer status (benign or malignant) based on input clinical features.

   **Request Body**:
   ```json
   {
     "age": 45,
     "tumor_size": 3.2,
     "family_history": "Yes",
     "tumor_type": "Invasive"
   }
   
### Flask App Workflow:

**Input Clinical Data**: Users provide clinical data such as age, tumor size, family history, etc.

**Prediction Output**: The model will output a prediction (benign or malignant) and the model's confidence score.

### Results

**Model Accuracy:** The best-performing model achieved an accuracy of 92%.

**Precision & Recall:** The model reached a precision of 91% and recall of 92%.

**AUC-ROC:** The final model achieved an AUC-ROC of 0.94, indicating high classification performance.
