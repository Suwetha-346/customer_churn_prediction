import pandas as pd
import pygwalker as pyg
import os

def generate_dashboard():
    print("Generating Interactive Power BI style dashboard...")
    
    # Load the original dataset
    df = pd.read_csv('dataset/telco_customer_churn.csv')
    
    # Ensure outputs folder exists
    os.makedirs('outputs', exist_ok=True)
    
    # Generate standalone HTML
    html_content = pyg.to_html(df)
    
    output_path = os.path.join('outputs', 'interactive_dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Dashboard successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_dashboard()
