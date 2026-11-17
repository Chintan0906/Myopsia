"""
Model Training and Persistence Pipeline
Project: AI-Based Eye Disease Prediction and Clinical Decision Support System
"""

import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    load_dataset,
    prepare_train_test_split,
    build_preprocessor,
    ALL_PREDICTIVE_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    TARGET_COLUMN
)


def train_and_evaluate_all_models(X_train, X_test, y_train, y_test) -> dict:
    """Trains multiple candidate models to establish baselines and benchmark comparisons."""
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision_Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=150, max_depth=14, class_weight='balanced', random_state=42)
    }
    
    benchmark_metrics = {}
    
    print("\n" + "=" * 65)
    print("BENCHMARKING CANDIDATE MODELS")
    print("=" * 65)
    
    for name, clf in models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        macro_prec = precision_score(y_test, y_pred, average='macro')
        macro_rec = recall_score(y_test, y_pred, average='macro')
        
        benchmark_metrics[name] = {
            'accuracy': round(float(acc), 4),
            'macro_f1': round(float(macro_f1), 4),
            'macro_precision': round(float(macro_prec), 4),
            'macro_recall': round(float(macro_rec), 4)
        }
        
        print(f"[{name.replace('_', ' ')}] Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%")
        
    return benchmark_metrics


def train_champion_pipeline(X_train, y_train) -> Pipeline:
    """Builds and fits the primary champion Random Forest pipeline."""
    preprocessor = build_preprocessor()
    champion_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', champion_clf)
    ])
    
    print("\nFitting Champion Random Forest Pipeline...")
    pipeline.fit(X_train, y_train)
    print("Champion Pipeline successfully fitted.")
    return pipeline


def export_model_artifacts(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    benchmark_results: dict,
    output_dir: str = 'saved_models'
):
    """Saves serialized pipeline, metadata JSON, and sample payload for web serving."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'eye_disease_pipeline.joblib')
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    sample_payload_path = os.path.join(output_dir, 'sample_patient_payload.json')
    
    # 1. Evaluate Champion
    y_pred = pipeline.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # 2. Serialize full scikit-learn Pipeline
    print(f"\nSerializing pipeline to: {model_path}")
    joblib.dump(pipeline, model_path, compress=3)
    
    # 3. Compile metadata
    classes = list(pipeline.classes_)
    metadata = {
        'model_name': 'Myopsia Random Forest Clinical Decision Support Classifier',
        'version': '1.0.0',
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'target_classes': classes,
        'features': {
            'all': ALL_PREDICTIVE_FEATURES,
            'categorical': CATEGORICAL_FEATURES,
            'binary': BINARY_FEATURES,
            'continuous': CONTINUOUS_FEATURES
        },
        'champion_metrics': {
            'test_accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'test_macro_f1': round(float(f1_score(y_test, y_pred, average='macro')), 4),
            'test_weighted_f1': round(float(f1_score(y_test, y_pred, average='weighted')), 4)
        },
        'per_class_metrics': {cls: report_dict[cls] for cls in classes if cls in report_dict},
        'all_models_benchmark': benchmark_results
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {metadata_path}")
    
    # 4. Generate representative sample patient JSON for testing web endpoints
    sample_patient = {
        "age": 68,
        "gender": "Female",
        "occupation": "Retired / Elderly",
        "diabetes_history": 0,
        "hypertension_history": 1,
        "family_history_eye_disease": 1,
        "previous_eye_surgery": 1,
        "previous_eye_injury": 0,
        "contact_lens_usage": 0,
        "smoking_history": "Former",
        
        "blurred_vision": 1,
        "black_spots_floaters": 0,
        "eye_pain": 0,
        "eye_redness": 0,
        "eye_itching": 0,
        "eye_dryness": 0,
        "excessive_watering": 0,
        "double_vision": 1,
        "flashes_of_light": 0,
        "light_sensitivity": 1,
        "difficulty_seeing_at_night": 1,
        "eye_swelling": 0,
        "headache": 0,
        "discharge_from_eye": 0,
        "foreign_body_sensation": 0,
        
        "symptom_duration_days": 240,
        "symptom_onset": "Gradual",
        "symptom_severity": "Moderate",
        "affected_eye": "Both",
        
        "intraocular_pressure_left": 15.5,
        "intraocular_pressure_right": 15.8,
        "visual_acuity_left": 0.65,
        "visual_acuity_right": 0.70,
        "cup_to_disc_ratio_left": 0.32,
        "cup_to_disc_ratio_right": 0.34,
        "family_history_glaucoma": 0,
        "previous_glaucoma_diagnosis": 0
    }
    
    with open(sample_payload_path, 'w', encoding='utf-8') as f:
        json.dump(sample_patient, f, indent=4)
    print(f"Saved sample patient payload to: {sample_payload_path}")


def main():
    print("=" * 65)
    print("STARTING END-TO-END MODEL TRAINING WORKFLOW")
    print("=" * 65)
    
    df = load_dataset('eye_disease_dataset.csv')
    print(f"Loaded {len(df)} patient records.")
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, test_size=0.20, random_state=42)
    print(f"Split completed: Train={len(X_train)}, Test={len(X_test)}")
    
    # 1. Benchmark all candidates
    benchmark_results = train_and_evaluate_all_models(X_train, X_test, y_train, y_test)
    
    # 2. Train Champion Pipeline
    champion_pipeline = train_champion_pipeline(X_train, y_train)
    
    # 3. Export Artifacts for Web Deployment
    export_model_artifacts(champion_pipeline, X_test, y_test, benchmark_results)
    
    print("\n" + "=" * 65)
    print("MODEL TRAINING & EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == '__main__':
    main()
