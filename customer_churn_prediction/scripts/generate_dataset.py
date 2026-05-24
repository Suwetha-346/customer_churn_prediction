import pandas as pd
import numpy as np
import os
import random

def generate_synthetic_telco_data(num_samples=7043, output_path="dataset/telco_customer_churn.csv"):
    """
    Generates a synthetic dataset modeled after the IBM Telco Customer Churn dataset.
    """
    np.random.seed(42)
    random.seed(42)

    print(f"Generating {num_samples} synthetic customer records...")

    # Define distributions and categories based on original dataset characteristics
    customer_id = [f"{random.randint(1000, 9999)}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}" for _ in range(num_samples)]
    
    gender = np.random.choice(['Male', 'Female'], num_samples)
    senior_citizen = np.random.choice([0, 1], num_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], num_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], num_samples, p=[0.30, 0.70])
    
    # Tenure: right skewed but generally uniform across some ranges, let's use a mix to simulate bimodal
    tenure_group1 = np.random.randint(1, 12, int(num_samples * 0.4))
    tenure_group2 = np.random.randint(12, 60, int(num_samples * 0.4))
    rem_samples = num_samples - len(tenure_group1) - len(tenure_group2)
    tenure_group3 = np.random.randint(60, 73, rem_samples)
    tenure = np.concatenate([tenure_group1, tenure_group2, tenure_group3])
    np.random.shuffle(tenure)
    
    phone_service = np.random.choice(['Yes', 'No'], num_samples, p=[0.90, 0.10])
    
    # Multiple lines depends on phone service
    multiple_lines = np.where(phone_service == 'No', 'No phone service', np.random.choice(['Yes', 'No'], num_samples, p=[0.45, 0.55]))
    
    internet_service = np.random.choice(['Fiber optic', 'DSL', 'No'], num_samples, p=[0.44, 0.34, 0.22])
    
    # Other services depend on internet service
    def internet_dependent_service(prob_yes):
        return np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], num_samples, p=[prob_yes, 1 - prob_yes]))

    online_security = internet_dependent_service(0.35)
    online_backup = internet_dependent_service(0.40)
    device_protection = internet_dependent_service(0.40)
    tech_support = internet_dependent_service(0.35)
    streaming_tv = internet_dependent_service(0.45)
    streaming_movies = internet_dependent_service(0.45)
    
    contract = np.random.choice(['Month-to-month', 'Two year', 'One year'], num_samples, p=[0.55, 0.24, 0.21])
    paperless_billing = np.random.choice(['Yes', 'No'], num_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], num_samples, p=[0.33, 0.23, 0.22, 0.22])
    
    # Monthly charges based on services
    monthly_charges = np.zeros(num_samples)
    monthly_charges += np.where(phone_service == 'Yes', 20, 0)
    monthly_charges += np.where(multiple_lines == 'Yes', 5, 0)
    monthly_charges += np.where(internet_service == 'DSL', 25, 0)
    monthly_charges += np.where(internet_service == 'Fiber optic', 50, 0)
    monthly_charges += np.where(online_security == 'Yes', 5, 0)
    monthly_charges += np.where(online_backup == 'Yes', 5, 0)
    monthly_charges += np.where(device_protection == 'Yes', 5, 0)
    monthly_charges += np.where(tech_support == 'Yes', 5, 0)
    monthly_charges += np.where(streaming_tv == 'Yes', 10, 0)
    monthly_charges += np.where(streaming_movies == 'Yes', 10, 0)
    # Add random noise
    monthly_charges += np.random.uniform(-5, 5, num_samples)
    monthly_charges = np.round(np.clip(monthly_charges, 18.25, 118.75), 2)
    
    # Total charges
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(-10, 10, num_samples), 2)
    total_charges_str = total_charges.astype(str)
    
    # Introduce some missing total charges for 0 tenure customers as in original dataset
    zero_tenure_mask = (tenure == 0)
    # Randomly set 11 samples to missing ' '
    missing_indices = np.random.choice(np.where(~zero_tenure_mask)[0], 11, replace=False)
    total_charges_str[missing_indices] = ' '
    
    # Simulate Churn based on logical factors
    churn_prob = np.zeros(num_samples)
    churn_prob += np.where(contract == 'Month-to-month', 0.4, 0.05)
    churn_prob += np.where(internet_service == 'Fiber optic', 0.15, 0.0)
    churn_prob += np.where(tech_support == 'No', 0.1, 0.0)
    churn_prob -= np.where(tenure > 24, 0.15, 0.0)
    churn_prob -= np.where(tenure > 48, 0.1, 0.0)
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    
    churn_num = np.random.binomial(1, churn_prob)
    churn = np.where(churn_num == 1, 'Yes', 'No')

    # Create DataFrame
    df = pd.DataFrame({
        'customerID': customer_id,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges_str,
        'Churn': churn
    })
    
    # Ensure dataset directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_telco_data()
