import pandas as pd
import joblib
import json
import argparse
import sys
import os

def predict(input_data):
    """
    Makes a prediction on raw input data.
    """
    models_dir = "models/"
    try:
        model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    except FileNotFoundError:
        print("Model artifacts not found. Please run train_models.py first.")
        sys.exit(1)

    df = pd.DataFrame([input_data])
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            # Handle unseen labels by defaulting to 0 or finding nearest
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError:
                df[col] = 0
                
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'Contract', 'PaymentMethod']
    
    # Since we use get_dummies in training, we need to ensure the exact same columns exist
    # Here we load the training columns to align them
    df = pd.get_dummies(df, columns=multi_cols)
    
    # Load model feature names if possible (RF/DT have feature_names_in_ usually)
    try:
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
    except AttributeError:
        print("Warning: Model does not have expected feature names. Prediction might fail if schema differs.")

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])

    # Predict
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]
    
    risk_tier = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
    
    return {
        "Churn_Prediction": "Yes" if pred == 1 else "No",
        "Churn_Probability": round(prob, 4),
        "Risk_Tier": risk_tier
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Customer Churn')
    parser.add_argument('--json', type=str, help='JSON string containing customer data')
    args = parser.parse_args()

    if args.json:
        customer_data = json.loads(args.json)
        result = predict(customer_data)
        print(json.dumps(result, indent=4))
    else:
        # Demo prediction
        sample_customer = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
        print("No JSON provided. Running on sample customer:")
        print(json.dumps(sample_customer, indent=4))
        result = predict(sample_customer)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=4))
