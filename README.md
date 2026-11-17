# Myopsia: AI-Based Eye Disease Prediction and Clinical Decision Support System Using Non-Image Clinical Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Educational / Synthetic](https://img.shields.io/badge/Status-Synthetic%20Educational%20Data-orange.svg)]()

> [!IMPORTANT]
> **ACADEMIC & CLINICAL DISCLAIMER:**
> This dataset has been generated **entirely synthetically** for college-level machine learning and data science research experimentation. 
> The records, physiological measurements, symptom complexes, and medical profiles **DO NOT** represent real clinical evidence, genuine human subjects, or medically validated diagnostic patterns.
> This repository must **NEVER** be used to train, evaluate, or deploy clinical diagnostic software for real-world patient care.

---

## 📌 Project Overview

**Myopsia** is an educational machine learning benchmark dataset and clinical decision support system (CDSS) prototype based strictly on **100% non-image clinical data**.

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

## 🔬 Missing Value Rationale

Selective missingness is intentionally incorporated to reflect authentic clinical triage protocols:
- **`cup_to_disc_ratio_left` / `cup_to_disc_ratio_right` (~39.5% missing):** Vertical CDR requires pharmacological pupillary dilation and slit-lamp biomicroscopy. In acute anterior infections (~78% omitted), ocular surface exams (~62% omitted), or refraction triage (~52% omitted), dilation is deferred or omitted.
- **`intraocular_pressure_left` / `intraocular_pressure_right` (~8.8% missing):** Tonometry is withheld during acute contagious conjunctivitis (~42% omitted) to prevent contact-prism contamination and viral cross-infection.
- **All other 35 variables:** 100% complete records (0% missing).

---

## 📁 Repository Structure

```text
Myopsia/
├── eye_disease_dataset.csv     # Primary dataset (5,000 records x 39 columns)
├── data_dictionary.csv         # Detailed metadata definition for all 39 columns
├── dataset_report.txt          # Technical report, methodology & distribution statistics
├── dataset_validation.ipynb    # Jupyter notebook with EDA, validation & ML benchmarks
├── generate_dataset.py         # Reproducible dataset generator script (NumPy / Pandas)
├── make_data_dictionary.py     # Data dictionary builder script
├── generate_report.py          # Summary report generation script
├── test_ml.py                  # Standalone multi-model ML evaluation script
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🚀 Machine Learning Benchmarks (80/20 Stratified Holdout)

Standard baseline models evaluated on holdout test set ($N=1,000$):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **95.7%** | **95.8%** | **94.1%** | **94.9%** |
| **Gradient Boosting Classifier** | **95.3%** | **95.2%** | **94.2%** | **94.7%** |
| **Logistic Regression (L2)** | **94.8%** | **94.5%** | **93.8%** | **94.1%** |
| **Decision Tree (Max Depth = 8)**| **91.0%** | **90.3%** | **90.0%** | **90.1%** |

---

## 🛠️ Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/Chintan0906/Myopsia.git
cd Myopsia
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Run validation and experiments
```bash
# Run standalone ML benchmark script
python test_ml.py

# Or launch Jupyter notebook
jupyter notebook dataset_validation.ipynb
```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
