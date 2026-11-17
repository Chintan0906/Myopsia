# Myopsia: AI-Based Eye Disease Prediction and Clinical Decision Support System Using Non-Image Clinical Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Educational / Synthetic](https://img.shields.io/badge/Status-Synthetic%20Educational%20Data-orange.svg)]()
[![Model: Random Forest](https://img.shields.io/badge/Champion%20Model-Random%20Forest%20(95.5%25)-green.svg)]()

> [!IMPORTANT]
> **ACADEMIC & CLINICAL DISCLAIMER:**
> This dataset has been generated **entirely synthetically** for college-level machine learning and data science research experimentation. 
> The records, physiological measurements, symptom complexes, and medical profiles **DO NOT** represent real clinical evidence, genuine human subjects, or medically validated diagnostic patterns.
> This repository must **NEVER** be used to train, evaluate, or deploy clinical diagnostic software for real-world patient care.

---

## 📌 Project Overview

**Myopsia** is an end-to-end machine learning system and clinical decision support system (CDSS) prototype based strictly on **100% non-image clinical data**.

While modern ophthalmology often relies on specialized imaging modalities such as Optical Coherence Tomography (OCT) and fundus photography, tertiary equipment is frequently unavailable in primary care, rural clinics, and low-resource triage environments. This project explores how multivariable machine learning algorithms (Random Forest, Gradient Boosting, Logistic Regression, Decision Trees) can predict ocular pathologies and prioritize triage using:
1. Patient demographic characteristics and vocational environment
2. Past systemic comorbidities and ocular surgical/injury history
3. Patient-reported symptom constellations, onset patterns, severity, and duration
4. Readily accessible non-image clinical parameters (Intraocular Pressure, Visual Acuity, Cup-to-Disc Ratio)

---

## 📊 Dataset Specifications

- **Total Patient Records:** Exactly **5,000 unique patient encounters** (`PID-00001` to `PID-05000`)
- **Total Features:** **38 predictive clinical/demographic features + 1 target variable**
- **Architecture:** 100% Structured Tabular Data (CSV format)
- **Target Variable:** `eye_disease` (7 multi-class conditions)
- **Class Distribution:** Moderate realistic imbalance reflecting outpatient ophthalmic prevalence:

| Diagnostic Class | Record Count | Proportion (%) | Typical Clinical Presentation |
| :--- | :---: | :---: | :--- |
| **Normal** | 950 | 19.0% | Routine wellness checkup, asymptomatic or mild screen fatigue |
| **Cataract** | 900 | 18.0% | Senile lens opacification, progressive blur, glare, night blindness |
| **Refractive Error** | 850 | 17.0% | Subnormal visual acuity, accommodative asthenopia, headache |
| **Dry Eye Disease** | 750 | 15.0% | Tear film instability, grittiness, redness, contact lens wear |
| **Glaucoma** | 600 | 12.0% | Optic neuropathy, cupping, elevated IOP (NTG overlap included) |
| **Conjunctivitis** | 550 | 11.0% | Acute anterior infection/allergy, discharge, hyperemia, itching |
| **Diabetic Retinopathy** | 400 | 8.0% | Microvascular retinal disease, diabetes comorbidity, floaters |
| **Total** | **5,000** | **100.0%** | |

---

## 📁 Repository Structure

```text
Myopsia/
├── app.py                            # Production web application & REST API (Flask)
├── requirements.txt                  # Python dependencies
├── eye_disease_dataset.csv           # Primary dataset (5,000 records x 39 columns)
├── data_dictionary.csv               # Detailed metadata definition for all 39 columns
├── dataset_report.txt                # Technical report, methodology & distribution statistics
│
├── saved_models/                     # Production model artifacts for web deployment
│   ├── eye_disease_pipeline.joblib   # Serialized end-to-end preprocessing + Random Forest pipeline
│   ├── model_metadata.json           # Target classes, schema, and benchmark metrics
│   ├── sample_patient_payload.json   # Ready-to-test JSON payload for API testing
│   ├── confusion_matrix.png          # Holdout test confusion matrix heatmap
│   └── feature_importance.png        # Top 15 clinical drivers plot
│
├── src/                              # Modular machine learning source code
│   ├── __init__.py
│   ├── data_preprocessing.py         # Data loading, cleaning, and ColumnTransformer pipeline
│   ├── train.py                      # Multi-model benchmarking & champion pipeline export
│   ├── evaluate.py                   # Holdout evaluation, confusion matrix & feature importance
│   └── inference.py                  # Production inference class (predicts class, probs & guidance)
│
├── templates/                        # Web dashboard templates
│   └── index.html                    # Responsive clinical UI with preset patient profiles
│
├── notebooks/                        # Educational & research Jupyter notebooks
│   ├── 01_data_cleaning_and_eda.ipynb           # Step-by-step cleaning, profiling, and EDA
│   └── 02_model_training_and_evaluation.ipynb  # Step-by-step training, benchmarking, and export
└── dataset_validation.ipynb          # Comprehensive validation & assertion checklist
```

---

## 🚀 Machine Learning Benchmarks (80/20 Stratified Holdout)

Standard baseline models evaluated on holdout test set ($N=1,000$):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Champion)** | **95.5%** | **95.3%** | **94.7%** | **94.8%** | Primary system engine (balanced class weights) |
| **Gradient Boosting** | **95.4%** | **95.1%** | **94.4%** | **94.8%** | Advanced boosting comparison |
| **Logistic Regression (L2)** | **95.2%** | **94.8%** | **94.4%** | **94.6%** | Linear statistical baseline |
| **Decision Tree (Max Depth = 8)**| **90.6%** | **89.9%** | **89.5%** | **89.7%** | Interpretable clinical flowchart |

---

## 🌐 Running the Web Application & REST API

The model is serialized as a self-contained scikit-learn `Pipeline` (including continuous imputation, scaling, categorical one-hot encoding, and the Random Forest classifier). It can be served directly via `app.py`.

### 1. Start the Web Server
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

### 2. Using the Interactive Web UI
- Use the **"Quick Test Profiles"** buttons at the top to load pre-configured patient profiles (*Routine Normal, Glaucoma Suspect, Senile Cataract, Diabetic Retinopathy, Acute Conjunctivitis, Dry Eye Disease*).
- Click **"Run AI Diagnostic Prediction"** to view:
  * The predicted eye disease condition.
  * Model confidence score & full probability distribution bar chart.
  * Color-coded clinical urgency rating (*High / Moderate / Routine*).
  * Recommended next steps (e.g., Humphrey perimetry, slit-lamp biomicroscopy, dilated fundus exam).

### 3. REST API Usage (`POST /api/predict`)
You can query the model from any website or backend using standard JSON:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d @saved_models/sample_patient_payload.json
```

**Sample API Response:**
```json
{
  "predicted_condition": "Cataract",
  "confidence_percentage": 97.28,
  "probability_distribution": {
    "Cataract": 97.28,
    "Diabetic_Retinopathy": 1.74,
    "Refractive_Error": 0.71,
    "Glaucoma": 0.19,
    "Dry_Eye_Disease": 0.08,
    "Conjunctivitis": 0.0,
    "Normal": 0.0
  },
  "clinical_guidance": {
    "triage_level": "Ophthalmic Outpatient Consultation",
    "urgency": "Moderate",
    "summary": "Age profile and gradual loss of visual clarity indicate senile crystalline lens opacification.",
    "next_steps": [
      "Slit-lamp biomicroscopy after pharmacological mydriasis.",
      "Assess functional impact on activities of daily living and driving.",
      "Plan for biometry and phacoemulsification with intraocular lens (IOL) implantation when indicated."
    ]
  }
}
```

---

## 🛠️ Python Inference in Custom Scripts

You can also load and use the model in any Python script with 3 lines of code:

```python
from src.inference import EyeDiseasePredictor

predictor = EyeDiseasePredictor()
result = predictor.predict({
    "age": 68,
    "blurred_vision": 1,
    "difficulty_seeing_at_night": 1,
    "symptom_duration_days": 240,
    "visual_acuity_left": 0.65,
    "visual_acuity_right": 0.70
})

print(result["predicted_condition"])   # e.g., 'Cataract'
print(result["confidence_percentage"]) # e.g., 97.28%
```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
