"""
Data Preprocessing and Pipeline Construction Module
Project: AI-Based Eye Disease Prediction and Clinical Decision Support System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Any

# Column Schema Definitions
CATEGORICAL_FEATURES = [
    'gender',
    'occupation',
    'smoking_history',
    'symptom_onset',
    'symptom_severity',
    'affected_eye'
]

BINARY_FEATURES = [
    'diabetes_history',
    'hypertension_history',
    'family_history_eye_disease',
    'previous_eye_surgery',
    'previous_eye_injury',
    'contact_lens_usage',
    'blurred_vision',
    'black_spots_floaters',
    'eye_pain',
    'eye_redness',
    'eye_itching',
    'eye_dryness',
    'excessive_watering',
    'double_vision',
    'flashes_of_light',
    'light_sensitivity',
    'difficulty_seeing_at_night',
    'eye_swelling',
    'headache',
    'discharge_from_eye',
    'foreign_body_sensation',
    'family_history_glaucoma',
    'previous_glaucoma_diagnosis'
]

CONTINUOUS_FEATURES = [
    'age',
    'symptom_duration_days',
    'intraocular_pressure_left',
    'intraocular_pressure_right',
    'visual_acuity_left',
    'visual_acuity_right',
    'cup_to_disc_ratio_left',
    'cup_to_disc_ratio_right'
]

NUMERICAL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES
ALL_PREDICTIVE_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'eye_disease'


def load_dataset(csv_path: str = 'eye_disease_dataset.csv') -> pd.DataFrame:
    """Load the raw synthetic eye disease clinical dataset."""
    df = pd.read_csv(csv_path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Perform data hygiene checks, type casting, and sanity validation."""
    df_clean = df.copy()
    
    # Ensure patient_id exists and is unique
    if 'patient_id' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['patient_id'])
    
    # Cast binary features to integer 0/1
    for col in BINARY_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            
    # Clip numerical features to physiological bounds
    if 'age' in df_clean.columns:
        df_clean['age'] = df_clean['age'].clip(18, 100)
    if 'symptom_duration_days' in df_clean.columns:
        df_clean['symptom_duration_days'] = df_clean['symptom_duration_days'].clip(0, 1000)
    if 'visual_acuity_left' in df_clean.columns:
        df_clean['visual_acuity_left'] = df_clean['visual_acuity_left'].clip(0.0, 2.0)
    if 'visual_acuity_right' in df_clean.columns:
        df_clean['visual_acuity_right'] = df_clean['visual_acuity_right'].clip(0.0, 2.0)
        
    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    1. Imputes missing continuous/binary features using median strategy + Standard Scaling.
    2. Imputes missing categorical features using most_frequent + OneHotEncoder.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def prepare_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates predictive features from target variable and executes
    a stratified train/test split.
    """
    df_clean = clean_dataset(df)
    X = df_clean[ALL_PREDICTIVE_FEATURES]
    y = df_clean[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = load_dataset()
    print(f"Loaded dataset with shape: {df.shape}")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    print(f"Transformed X_train shape: {X_train_trans.shape}")
