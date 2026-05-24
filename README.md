# Customer Churn Prediction Project

A complete end-to-end Machine Learning and Data Analytics project to predict customer churn in the Telecommunications sector. This project uses customer data to analyze churn behavior, perform exploratory data analysis (EDA), train multiple machine learning classification models, and deploy the best-performing model using an interactive Streamlit web application.

---

# Table of Contents

- Overview
- Features
- Technologies Used
- Project Structure
- Dataset Information
- Exploratory Data Analysis
- Machine Learning Models
- Model Evaluation
- Streamlit Web Application
- Installation
- How to Run the Project
- Sample Outputs
- Future Enhancements
- Author

---

# Overview

Customer churn prediction is one of the most important business problems in the telecommunications industry. Companies need to identify customers who are likely to leave their service so they can take preventive actions to improve customer retention.

This project helps:
- Analyze customer behavior
- Identify churn patterns
- Predict churn probability
- Visualize customer insights
- Support business decision-making

The project includes:
- Data preprocessing
- Data visualization
- Feature engineering
- Machine learning model training
- Model comparison
- Interactive dashboard deployment

---

#  Features

✅ Data Cleaning and Preprocessing  
✅ Exploratory Data Analysis (EDA)  
✅ Feature Engineering  
✅ Multiple ML Algorithms  
✅ Churn Prediction System  
✅ Interactive Streamlit Dashboard  
✅ Visualization Reports  
✅ Model Evaluation Metrics  
✅ Saved Trained Models  
✅ Professional Project Structure  

---

#  Technologies Used

## Programming Language
- Python

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

## Deployment
- Streamlit

## Development Tools
- Jupyter Notebook
- VS Code

---

#  Project Structure

```text
customer_churn_prediction/
├── dataset/
│   ├── telco_customer_churn.csv
│   ├── train_processed.csv
│   └── test_processed.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── outputs/
│   └── eda_plots/
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── scripts/
│   ├── generate_dataset.py
│   ├── data_preprocessing.py
│   ├── eda_visualization.py
│   ├── train_models.py
│   └── predict.py
│
├── app/
│   └── streamlit_app.py
│
└── requirements.txt
```

---

#  Dataset Information

The project uses the Telco Customer Churn dataset containing customer details such as:

- Gender
- Senior Citizen
- Tenure
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges
- Payment Method
- Churn Status

### Target Variable
- `Churn`
  - Yes → Customer leaves service
  - No → Customer stays

---

# 📈 Exploratory Data Analysis (EDA)

The project performs detailed exploratory data analysis to identify important churn trends.

## Visualizations Included

- Churn Distribution
- Contract Type Analysis
- Monthly Charges Distribution
- Tenure Analysis
- Internet Service Analysis
- Correlation Heatmap
- Customer Demographics Analysis

### Example Insights
- Customers with month-to-month contracts have higher churn rates.
- Customers with higher monthly charges are more likely to churn.
- Long-term customers are less likely to leave.

---

#  Machine Learning Models

The following classification algorithms are implemented:

## 1. Logistic Regression
- Simple and interpretable model
- Good baseline classifier

## 2. Decision Tree
- Easy to visualize
- Handles nonlinear relationships

## 3. Random Forest
- Ensemble learning technique
- High accuracy and stability

## 4. XGBoost
- Advanced boosting algorithm
- Better predictive performance

---

# Model Evaluation

Models are evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

The best-performing model is saved for deployment.

---

#  Streamlit Web Application

An interactive Streamlit dashboard is included for real-time customer churn prediction.

## Features
- User-friendly interface
- Customer data input form
- Real-time prediction
- Churn probability visualization
- Business insights dashboard

---

#  Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/customer_churn_prediction.git
cd customer_churn_prediction
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows
```bash
venv\Scripts\activate
```

### Linux/Mac
```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  How to Run the Project

## Step 1: Train Models

```bash
python scripts/train_models.py
```

## Step 2: Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

# 📸 Sample Outputs

## EDA Charts
- Customer churn distribution graphs
- Correlation heatmaps
- Contract analysis charts

## Prediction Results
- Churn Probability
- Risk Classification
- Customer Insights

---

#  Future Enhancements

- Deep Learning Integration
- Real-time Database Support
- Cloud Deployment
- API Development
- Automated Retraining Pipeline
- Advanced Customer Segmentation

---

#  Business Impact

This project helps companies:
- Reduce customer loss
- Improve customer retention
- Increase revenue
- Understand customer behavior
- Make data-driven decisions

---

## Skills
- Python
- SQL
- Power BI
- Machine Learning
- Data Analytics


