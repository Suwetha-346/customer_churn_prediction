import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

def train_and_evaluate(data_dir="dataset/", models_dir="models/", outputs_dir="outputs/"):
    print("Starting Model Training...")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))

    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    # Calculate positive weight for XGBoost
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    metrics = []
    trained_models = {}
    conf_matrices = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        trained_models[name] = model
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        
        print(f"--- {name} Performance ---")
        print(classification_report(y_test, y_pred))

    metrics_df = pd.DataFrame(metrics)
    print("\nModel Comparison:")
    print(metrics_df)

    # Save Best Model (based on F1-Score)
    best_model_name = metrics_df.loc[metrics_df['F1-Score'].idxmax()]['Model']
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    print(f"\nBest Model: {best_model_name} saved to {models_dir}/best_model.pkl")

    # Plot Model Comparison
    metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='viridis')
    plt.title('Model Performance Comparison', fontsize=14)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(outputs_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (name, cm) in enumerate(conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Feature Importance (if best model is Random Forest or Decision Tree)
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(10, 8))
        importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
        importances.nlargest(15).sort_values().plot(kind='barh', color='teal')
        plt.title(f'Top 15 Feature Importances ({best_model_name})', fontsize=14)
        plt.savefig(os.path.join(outputs_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Saved all training outputs and charts.")

if __name__ == "__main__":
    train_and_evaluate()
