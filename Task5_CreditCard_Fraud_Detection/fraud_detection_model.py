# fraud_detection_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
try:
    df = pd.read_csv("creditcard.csv")
except FileNotFoundError:
    print("ERROR: creditcard.csv file not found. Please make sure it's in the same folder.")
    exit()

# Drop 'Time' column
if 'Time' in df.columns:
    df.drop('Time', axis=1, inplace=True)

# Normalize 'Amount' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr_model.predict(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)

# Evaluate and write to text file
with open("output.txt", "w") as f:
    f.write("=== Original Class Distribution ===\n")
    f.write(str(y.value_counts()) + "\n\n")

    f.write("=== Resampled Class Distribution ===\n")
    f.write(str(y_train_resampled.value_counts()) + "\n\n")

    f.write("=== Logistic Regression Classification Report ===\n")
    f.write(classification_report(y_test, y_pred_lr))
    f.write("\n")

    f.write("=== Random Forest Classification Report ===\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write("\n")

    conf_matrix = confusion_matrix(y_test, y_pred_rf)
    f.write("=== Confusion Matrix (Random Forest) ===\n")
    f.write(str(conf_matrix))
