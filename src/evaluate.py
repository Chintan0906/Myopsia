"""
Model Evaluation and Diagnostic Reporting Module
Project: AI-Based Eye Disease Prediction and Clinical Decision Support System
"""

import os
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.data_preprocessing import (
    load_dataset,
    prepare_train_test_split,
    ALL_PREDICTIVE_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)


def evaluate_pipeline(
    model_path: str = 'saved_models/eye_disease_pipeline.joblib',
    data_path: str = 'eye_disease_dataset.csv',
    output_dir: str = 'saved_models'
):
    """Loads saved pipeline and runs comprehensive diagnostic evaluation on holdout test set."""
    print("=" * 65)
    print("RUNNING MODEL EVALUATION & GENERATING PERFORMANCE PLOTS")
    print("=" * 65)
    
    # 1. Load Model & Data
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}. Run src/train.py first.")
        
    pipeline = joblib.load(model_path)
    df = load_dataset(data_path)
    _, X_test, _, y_test = prepare_train_test_split(df, test_size=0.20, random_state=42)
    
    classes = list(pipeline.classes_)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n[Overall Performance]")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Macro F1:    {macro_f1*100:.2f}%")
    print(f"Weighted F1: {weighted_f1*100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Holdout Confusion Matrix (N = 1,000)', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Diagnostic Class', fontsize=11)
    plt.ylabel('True Diagnostic Class', fontsize=11)
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")
    
    # 3. Feature Importance Plot
    rf_clf = pipeline.named_steps['clf']
    prep = pipeline.named_steps['prep']
    
    cat_encoder = prep.named_transformers_['cat'].named_steps['onehot']
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERICAL_FEATURES + cat_names
    
    importances = pd.Series(rf_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 7))
    importances.head(15).iloc[::-1].plot(kind='barh', color='#2980b9')
    plt.title('Top 15 Most Important Features in Clinical Decision Pipeline', fontsize=13, fontweight='bold')
    plt.xlabel('Gini Importance (Mean Decrease in Impurity)', fontsize=11)
    plt.ylabel('Clinical / Demographic Feature', fontsize=11)
    plt.tight_layout()
    
    fi_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to: {fi_path}")


if __name__ == '__main__':
    evaluate_pipeline()
