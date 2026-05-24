import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import pygwalker as pyg
import streamlit.components.v1 as components

# Configuration
st.set_page_config(page_title="Customer Churn Analysis", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for aesthetics
st.markdown("""
<style>
    /* Premium Glassmorphism Theme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main {
        background-color: transparent;
    }
    h1, h2, h3, p, span, div, label {
        color: #000000 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4361EE 0%, #3A0CA3 100%); 
        color: white !important; 
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.4);
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.85); 
        backdrop-filter: blur(10px);
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1); 
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
        color: #000000 !important;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #4361EE !important;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #333333 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    try:
        model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
        return model, scaler, encoders
    except Exception as e:
        return None, None, None

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'dataset', 'telco_customer_churn.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df
    return None

model, scaler, encoders = load_models()
df = load_data()

st.title("📊 Customer Churn Dashboard & Analysis")

if df is not None:
    # Sidebar Filtering
    st.sidebar.header("🔍 Dashboard Filters")
    st.sidebar.write("Filter the dataset for analysis:")
    
    contract_filter = st.sidebar.multiselect("Contract Type", options=df['Contract'].unique(), default=df['Contract'].unique())
    internet_filter = st.sidebar.multiselect("Internet Service", options=df['InternetService'].unique(), default=df['InternetService'].unique())
    payment_filter = st.sidebar.multiselect("Payment Method", options=df['PaymentMethod'].unique(), default=df['PaymentMethod'].unique())
    tenure_filter = st.sidebar.slider("Tenure Range (Months)", min_value=int(df['tenure'].min()), max_value=int(df['tenure'].max()), value=(int(df['tenure'].min()), int(df['tenure'].max())))
    
    # Apply Filters
    filtered_df = df[
        (df['Contract'].isin(contract_filter)) &
        (df['InternetService'].isin(internet_filter)) &
        (df['PaymentMethod'].isin(payment_filter)) &
        (df['tenure'] >= tenure_filter[0]) & (df['tenure'] <= tenure_filter[1])
    ]

    tab_pbi, tab1, tab3 = st.tabs(["🔀 Power BI Explorer", "📊 Power BI Style Dashboard", "🔮 Predict Churn"])

    with tab_pbi:
        st.header("Power BI / Tableau Style Data Explorer")
        
        @st.cache_data
        def get_pyg_html(dataframe):
            # Using light theme to match the rest of the app and look professional
            return pyg.to_html(dataframe, themeKey='light', dark='light')
            
        components.html(get_pyg_html(df), height=900, scrolling=True)

    with tab1:
        st.header("Dataset Analysis & Key Metrics")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        total_customers = len(filtered_df)
        churn_rate = (filtered_df['Churn'] == 'Yes').mean() * 100 if total_customers > 0 else 0
        avg_tenure = filtered_df['tenure'].mean() if total_customers > 0 else 0
        avg_monthly = filtered_df['MonthlyCharges'].mean() if total_customers > 0 else 0
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Customers</div><div class="metric-value">{total_customers:,}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Churn Rate</div><div class="metric-value" style="color: {"#F72585" if churn_rate > 25 else "#4361EE"};">{churn_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Tenure</div><div class="metric-value">{avg_tenure:.1f} mo</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Monthly Revenue</div><div class="metric-value">${avg_monthly:.2f}</div></div>', unsafe_allow_html=True)

        if total_customers > 0:
            st.markdown("---")
            
            # 1. Churn Distribution
            st.subheader("Churn Distribution")
            fig1 = px.pie(filtered_df, names='Churn', hole=0.4, color_discrete_sequence=['#4361EE', '#F72585'])
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)
            
            # 2. Monthly Charges Distribution
            st.markdown("---")
            st.subheader("Monthly Charges Distribution")
            fig2 = px.box(filtered_df, x='Churn', y='MonthlyCharges', color='Churn', color_discrete_sequence=['#4361EE', '#F72585'])
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
            
            # 3. Churn by Contract Type
            st.markdown("---")
            st.subheader("Churn by Contract Type")
            fig3 = px.histogram(filtered_df, x='Contract', color='Churn', barmode='group', color_discrete_sequence=['#4361EE', '#F72585'])
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
            
            # 4. Churn by Internet Service
            st.markdown("---")
            st.subheader("Churn by Internet Service")
            fig4 = px.histogram(filtered_df, x='InternetService', color='Churn', barmode='group', color_discrete_sequence=['#4361EE', '#F72585'])
            fig4.update_layout(height=500)
            st.plotly_chart(fig4, use_container_width=True)
            
            # 5. Charges vs Tenure Scatter
            st.markdown("---")
            st.subheader("Charges vs Tenure (Sampled)")
            fig5 = px.scatter(filtered_df.sample(min(1000, len(filtered_df))), x='tenure', y='MonthlyCharges', color='Churn', color_discrete_sequence=['#4361EE', '#F72585'], opacity=0.7)
            fig5.update_layout(height=500)
            st.plotly_chart(fig5, use_container_width=True)

            # 6. Churn by Payment Method
            st.markdown("---")
            st.subheader("Churn by Payment Method")
            fig6 = px.histogram(filtered_df, y='PaymentMethod', color='Churn', barmode='stack', color_discrete_sequence=['#4361EE', '#F72585'], orientation='h')
            fig6.update_layout(height=500)
            st.plotly_chart(fig6, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Raw Data Explorer")
            st.write("Browse the filtered dataset:")
            st.dataframe(filtered_df.head(100), use_container_width=True)
        else:
            st.warning("No data matches the selected filters. Please adjust the sidebar filters.")

    with tab3:
        st.header("Predict Single Customer Churn")
        if model is None:
            st.error("Model artifacts not found. Please train models first.")
        else:
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    senior = st.selectbox("Senior Citizen", [0, 1])
                    partner = st.selectbox("Partner", ["Yes", "No"])
                    dependents = st.selectbox("Dependents", ["Yes", "No"])
                    tenure = st.slider("Tenure (Months)", 0, 72, 1)
                    
                with col2:
                    phone = st.selectbox("Phone Service", ["Yes", "No"])
                    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                    
                with col3:
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                    monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=50.0)
                    total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=50.0)
                
                submit = st.form_submit_button("Predict Churn Risk")
                
                if submit:
                    # Preprocess single record
                    input_data = {
                        "gender": gender, "SeniorCitizen": senior, "Partner": partner, 
                        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, 
                        "MultipleLines": multiple, "InternetService": internet, 
                        "OnlineSecurity": security, "OnlineBackup": backup, 
                        "DeviceProtection": "No", "TechSupport": "No", 
                        "StreamingTV": "No", "StreamingMovies": "No", 
                        "Contract": contract, "PaperlessBilling": paperless, 
                        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total
                    }
                    
                    df_input = pd.DataFrame([input_data])
                    
                    # Encode binary
                    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
                    for col in binary_cols:
                        df_input[col] = encoders[col].transform(df_input[col])
                    
                    # Encode multi-class
                    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                  'Contract', 'PaymentMethod']
                    df_input = pd.get_dummies(df_input, columns=multi_cols)
                    
                    # Align columns
                    for col in model.feature_names_in_:
                        if col not in df_input.columns:
                            df_input[col] = 0
                    df_input = df_input[model.feature_names_in_]
                    
                    # Scale
                    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
                    df_input[num_cols] = scaler.transform(df_input[num_cols])
                    
                    # Predict
                    prob = model.predict_proba(df_input)[0][1]
                    pred = model.predict(df_input)[0]
                    
                    st.markdown("---")
                    st.subheader("Prediction Result")
                    res_col1, res_col2 = st.columns(2)
                    
                    if pred == 1:
                        res_col1.error(f"⚠️ High Risk of Churn")
                    else:
                        res_col1.success(f"✅ Likely to Stay")
                        
                    res_col2.metric("Churn Probability", f"{prob*100:.2f}%")

else:
    st.error("Dataset not found. Please ensure the data generation script has been run.")
