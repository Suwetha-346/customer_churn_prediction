import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path="dataset/telco_customer_churn.csv", output_dir="dataset/", models_dir="models/"):
    """
    Cleans, encodes, and scales the dataset.
    Splits into train and test sets and saves them.
    """
    print("Starting data preprocessing...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    # Handle TotalCharges: Convert to numeric, errors to NaN, fill with median
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Just in case any other NaNs crept in
    df = df.fillna(0)

    # Drop customerID as it's not a useful feature
    df.drop(columns=['customerID'], inplace=True)

    # Separate feature types
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'Contract', 'PaymentMethod']
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Label Encoding for binary columns
    encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # One-Hot Encoding for multi-categorical columns
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Split features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Train-test split (80-20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numeric features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Save artifacts
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(models_dir, 'label_encoders.pkl'))
    print("Saved scaler and label encoders.")

    # Save processed splits
    X_train['Churn'] = y_train
    X_test['Churn'] = y_test
    X_train.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
    print("Preprocessing complete. Saved train and test datasets.")

if __name__ == "__main__":
    preprocess_data()
