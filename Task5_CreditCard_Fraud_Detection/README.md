# Task 5 - Credit Card Fraud Detection 💳

This is Task 5 of the CODSOFT Data Science Internship.

## 📁 Dataset
- *creditcard.csv*  
  The dataset contains:
  - Transaction details transformed via PCA
  - Normalized 'Amount' field
  - 'Class' column (0 = Genuine, 1 = Fraud)

> ⚠ Dataset source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
>  
> Dataset not included in this repo due to size and licensing. Please download manually if needed.

## 📊 Goal
Detect fraudulent credit card transactions using classification models with imbalanced data handling.

## 📌 Files Included
- fraud_detection_model.py — Python script for data preprocessing, model training, and evaluation.
- output.txt — Contains precision, recall, F1-score, and confusion matrix for both models.

## 🧠 Models Used
- Logistic Regression
- Random Forest Classifier

## 🧪 Techniques Applied
- Data Normalization
- Train-test splitting (stratified)
- *SMOTE* for class imbalance handling

## 📌 Output
Evaluation metrics (precision, recall, F1-score) and confusion matrix are saved in output.txt.

---
