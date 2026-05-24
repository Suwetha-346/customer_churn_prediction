# 🔮 Customer Churn Prediction Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green.svg)

A complete, end-to-end Machine Learning project to predict customer churn in the Telecommunications sector. This project takes raw customer data, performs Exploratory Data Analysis (EDA), trains multiple classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost), and deploys the best performing model via an interactive Streamlit web application.

## 📂 Project Structure

```text
customer_churn_prediction/
├── dataset/
│   ├── telco_customer_churn.csv        # Raw synthetic dataset based on IBM Telco
│   ├── train_processed.csv             # Cleaned and scaled training data
│   └── test_processed.csv              # Cleaned and scaled test data
├── models/
│   ├── best_model.pkl                  # Saved Random Forest / LR model
│   ├── scaler.pkl                      # Saved StandardScaler
│   └── label_encoders.pkl              # Saved categorical encoders
├── outputs/
│   └── eda_plots/                      # Generated visualization PNGs
├── notebooks/
│   └── customer_churn_analysis.ipynb   # Full interactive Jupyter notebook
├── scripts/
│   ├── generate_dataset.py             # Script to generate synthetic Telco data
│   ├── data_preprocessing.py           # Data cleaning & feature engineering script
│   ├── eda_visualization.py            # Automated exploratory data analysis script
│   ├── train_models.py                 # Model training & evaluation script
│   └── predict.py                      # CLI tool for single customer predictions
├── app/
│   └── streamlit_app.py                # Interactive web dashboard and prediction tool
└── requirements.txt                    # Project dependencies
```

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Run Pipeline
To run the entire pipeline from scratch, execute the scripts in order:

```bash
# Generate the synthetic dataset (matches IBM Telco dataset schema)
python scripts/generate_dataset.py

# Preprocess data (Encoding, Scaling, Train-Test Split)
python scripts/data_preprocessing.py

# Generate EDA Visualizations (saved to outputs/eda_plots/)
python scripts/eda_visualization.py

# Train models, evaluate, and save the best one
python scripts/train_models.py
```

### 3. Launch the Web Application
Run the Streamlit dashboard to explore data, view model performance, and make real-time predictions:
```bash
streamlit run app/streamlit_app.py
```

### 4. CLI Prediction
You can also predict customer churn directly from the command line:
```bash
python scripts/predict.py
```

## 📊 Business Insights

Based on the model and EDA:
- **Contract Type**: Month-to-month contracts have the highest churn rate. Incentivizing 1-year or 2-year contracts drastically improves retention.
- **Tenure**: Customers are most likely to churn within their first 6 months. Implementing a strong onboarding program is crucial.
- **Internet Service**: Fiber optic customers show a higher churn rate compared to DSL, potentially indicating pricing or reliability concerns that need business review.

## 🚀 Deployment Support

### Option 1: Docker
A `Dockerfile` is included for easy containerization.
```bash
# Build the Docker image
docker build -t customer-churn-app .

# Run the container
docker run -p 8501:8501 customer-churn-app
```
Navigate to `http://localhost:8501` to view the app.

### Option 2: Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and deploy a new app.
3. Select `app/streamlit_app.py` as the main file path.
4. Streamlit will automatically install the packages in `requirements.txt` and host your project for free!

## 🛠️ Built With
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, XGBoost
* **Visualization**: Matplotlib, Seaborn, Plotly, PyGWalker (Power BI Explorer)
* **Deployment/App**: Streamlit, Docker
