import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

def generate_eda_plots(input_path="dataset/telco_customer_churn.csv", output_dir="outputs/eda_plots/"):
    print("Starting EDA visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    colors = ["#4361EE", "#F72585"]
    
    # 1. Churn Distribution
    plt.figure(figsize=(6, 6))
    churn_counts = df['Churn'].value_counts()
    plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=colors, startangle=90, explode=[0, 0.1])
    plt.title('Customer Churn Distribution', fontsize=14)
    plt.savefig(os.path.join(output_dir, '1_churn_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly Charges Analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, palette=colors, bins=30)
    plt.title('Monthly Charges Distribution by Churn', fontsize=14)
    plt.savefig(os.path.join(output_dir, '2_monthly_charges_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Contract Type Analysis
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Contract', hue='Churn', palette=colors)
    plt.title('Churn Rate by Contract Type', fontsize=14)
    plt.savefig(os.path.join(output_dir, '3_contract_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Internet Service Analysis
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='InternetService', hue='Churn', palette=colors)
    plt.title('Churn Rate by Internet Service', fontsize=14)
    plt.savefig(os.path.join(output_dir, '4_internet_service_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Tenure Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['Churn'] == 'No'], x='tenure', fill=True, color=colors[0], label='No Churn')
    sns.kdeplot(data=df[df['Churn'] == 'Yes'], x='tenure', fill=True, color=colors[1], label='Churn')
    plt.title('Customer Tenure Distribution', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(output_dir, '5_tenure_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Correlation Heatmap (Numeric features only for clarity)
    plt.figure(figsize=(8, 6))
    numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
    numeric_df['Churn_Label'] = df['Churn'].map({'Yes': 1, 'No': 0})
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap', fontsize=14)
    plt.savefig(os.path.join(output_dir, '6_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved 6 EDA plots to {output_dir}")

if __name__ == "__main__":
    generate_eda_plots()
