import numpy as np
import pandas as pd
import os

def generate_eye_disease_dataset(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    
    classes = [
        'Normal',
        'Cataract',
        'Refractive_Error',
        'Dry_Eye_Disease',
        'Glaucoma',
        'Conjunctivitis',
        'Diabetic_Retinopathy'
    ]
    
    # Moderate realistic class imbalance summing to exactly 5,000
    class_counts = {
        'Normal': 950,
        'Cataract': 900,
        'Refractive_Error': 850,
        'Dry_Eye_Disease': 750,
        'Glaucoma': 600,
        'Conjunctivitis': 550,
        'Diabetic_Retinopathy': 400
    }
    
    occupations = [
        'Office / Desk Worker',
        'Manual Laborer / Construction',
        'Driver / Transport',
        'Healthcare Professional',
        'Teacher / Academic',
        'Farmer / Agricultural',
        'Retired / Elderly',
        'Student',
        'Homemaker',
        'Sales / Retail'
    ]
    
    records = []
    current_id = 1
    
    for cls in classes:
        count = class_counts[cls]
        
        for _ in range(count):
            p_id = f"PID-{current_id:05d}"
            current_id += 1
            
            # --- Demographics & History ---
            if cls == 'Normal':
                age = int(np.clip(np.random.normal(41, 14), 18, 78))
                gender = np.random.choice(['Male', 'Female'], p=[0.49, 0.51])
                occ_p = [0.22, 0.10, 0.08, 0.08, 0.10, 0.08, 0.08, 0.12, 0.08, 0.06]
                diabetes = int(np.random.rand() < 0.09)
                hypertension = int(np.random.rand() < 0.22)
                fam_eye = int(np.random.rand() < 0.18)
                prev_surg = int(np.random.rand() < 0.04)
                prev_inj = int(np.random.rand() < 0.03)
                contact_lens = int(np.random.rand() < 0.14)
                smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.66, 0.20, 0.14])
                
                # 35% of normal patients presenting for routine checkup report mild digital eyestrain/dryness
                has_mild_complaint = (np.random.rand() < 0.35)
                
                blurred_vis = int(has_mild_complaint and np.random.rand() < 0.25)
                floaters = int(np.random.rand() < 0.05)
                eye_pain = int(has_mild_complaint and np.random.rand() < 0.15)
                eye_redness = int(has_mild_complaint and np.random.rand() < 0.22)
                eye_itch = int(np.random.rand() < 0.12)
                eye_dry = int(has_mild_complaint and np.random.rand() < 0.32)
                watering = int(has_mild_complaint and np.random.rand() < 0.15)
                diplopia = 0
                flashes = 0
                photophobia = int(has_mild_complaint and np.random.rand() < 0.10)
                night_blind = int(np.random.rand() < 0.04)
                swelling = 0
                headache = int(has_mild_complaint and np.random.rand() < 0.35)
                discharge = 0
                foreign_body = int(has_mild_complaint and np.random.rand() < 0.08)
                
                if has_mild_complaint and (blurred_vis or eye_dry or headache or eye_redness or eye_itch):
                    symptom_dur = int(np.random.randint(2, 28))
                    symptom_onset = np.random.choice(['Gradual', 'Sudden'], p=[0.88, 0.12])
                    symptom_sev = 'Mild'
                    affected_eye = np.random.choice(['Both', 'Left', 'Right'], p=[0.75, 0.13, 0.12])
                else:
                    symptom_dur = 0
                    symptom_onset = 'None'
                    symptom_sev = 'None'
                    affected_eye = 'None'
                    
                # Clinical
                iop_base = np.random.normal(15.4, 2.5)
                if np.random.rand() < 0.04:
                    iop_base += np.random.uniform(5.5, 8.0)
                iop_l = round(float(np.clip(iop_base + np.random.normal(0, 0.6), 10.0, 24.5)), 1)
                iop_r = round(float(np.clip(iop_base + np.random.normal(0, 0.6), 10.0, 24.5)), 1)
                
                va_l = round(float(np.clip(np.random.normal(0.02, 0.03), 0.00, 0.12)), 2)
                va_r = round(float(np.clip(np.random.normal(0.02, 0.03), 0.00, 0.12)), 2)
                
                cdr_base = np.random.normal(0.29, 0.06)
                if np.random.rand() < 0.05:
                    cdr_base += 0.20
                cdr_l = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.15, 0.55)), 2)
                cdr_r = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.15, 0.55)), 2)
                
                fam_glauc = int(np.random.rand() < 0.07)
                prev_glauc = int(np.random.rand() < 0.01)
                
                cdr_missing = (np.random.rand() < 0.35)
                iop_missing = (np.random.rand() < 0.04)
                
            elif cls == 'Cataract':
                age = int(np.clip(np.random.normal(67, 8), 48, 88))
                gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
                occ_p = [0.06, 0.08, 0.06, 0.04, 0.06, 0.16, 0.42, 0.00, 0.08, 0.04]
                diabetes = int(np.random.rand() < 0.28)
                hypertension = int(np.random.rand() < 0.50)
                fam_eye = int(np.random.rand() < 0.34)
                prev_surg = int(np.random.rand() < 0.20)
                prev_inj = int(np.random.rand() < 0.06)
                contact_lens = int(np.random.rand() < 0.02)
                smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.48, 0.34, 0.18])
                
                is_early = (np.random.rand() < 0.08)
                
                blurred_vis = int(not is_early or np.random.rand() < 0.40)
                floaters = int(np.random.rand() < 0.16)
                eye_pain = int(np.random.rand() < 0.04)
                eye_redness = int(np.random.rand() < 0.04)
                eye_itch = int(np.random.rand() < 0.05)
                eye_dry = int(np.random.rand() < 0.20)
                watering = int(np.random.rand() < 0.16)
                diplopia = int(np.random.rand() < 0.20)
                flashes = int(np.random.rand() < 0.05)
                photophobia = int(not is_early and np.random.rand() < 0.62)
                night_blind = int(not is_early and np.random.rand() < 0.70)
                swelling = 0
                headache = int(np.random.rand() < 0.12)
                discharge = 0
                foreign_body = int(np.random.rand() < 0.06)
                
                if is_early and not (blurred_vis or photophobia or night_blind):
                    symptom_dur = 0
                    symptom_onset = 'None'
                    symptom_sev = 'None'
                    affected_eye = 'None'
                else:
                    symptom_dur = int(np.clip(np.random.normal(300, 140), 30, 730))
                    symptom_onset = np.random.choice(['Gradual', 'Sudden'], p=[0.96, 0.04])
                    symptom_sev = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.28, 0.50, 0.22])
                    affected_eye = np.random.choice(['Both', 'Left', 'Right'], p=[0.68, 0.16, 0.16])
                
                iop_base = np.random.normal(15.6, 2.5)
                iop_l = round(float(np.clip(iop_base + np.random.normal(0, 0.7), 10.0, 22.0)), 1)
                iop_r = round(float(np.clip(iop_base + np.random.normal(0, 0.7), 10.0, 22.0)), 1)
                
                va_base = np.random.normal(0.52, 0.20)
                va_l = round(float(np.clip(va_base + np.random.normal(0, 0.10), 0.15, 1.20)), 2)
                va_r = round(float(np.clip(va_base + np.random.normal(0, 0.10), 0.15, 1.20)), 2)
                
                cdr_base = np.random.normal(0.32, 0.07)
                cdr_l = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.18, 0.52)), 2)
                cdr_r = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.18, 0.52)), 2)
                
                fam_glauc = int(np.random.rand() < 0.08)
                prev_glauc = int(np.random.rand() < 0.02)
                
                cdr_missing = (np.random.rand() < 0.26)
                iop_missing = (np.random.rand() < 0.04)

            elif cls == 'Refractive_Error':
                age = int(np.clip(np.random.normal(31, 10), 18, 65))
                gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
                occ_p = [0.26, 0.08, 0.06, 0.08, 0.10, 0.04, 0.02, 0.24, 0.06, 0.06]
                diabetes = int(np.random.rand() < 0.07)
                hypertension = int(np.random.rand() < 0.15)
                fam_eye = int(np.random.rand() < 0.44)
                prev_surg = int(np.random.rand() < 0.05)
                prev_inj = int(np.random.rand() < 0.03)
                contact_lens = int(np.random.rand() < 0.38)
                smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.70, 0.17, 0.13])
                
                blurred_vis = int(np.random.rand() < 0.92)
                floaters = int(np.random.rand() < 0.14)
                eye_pain = int(np.random.rand() < 0.12)
                eye_redness = int(np.random.rand() < 0.14)
                eye_itch = int(np.random.rand() < 0.12)
                eye_dry = int(np.random.rand() < 0.22)
                watering = int(np.random.rand() < 0.14)
                diplopia = int(np.random.rand() < 0.15)
                flashes = int(np.random.rand() < 0.04)
                photophobia = int(np.random.rand() < 0.18)
                night_blind = int(np.random.rand() < 0.35)
                swelling = 0
                headache = int(np.random.rand() < 0.62)
                discharge = 0
                foreign_body = int(np.random.rand() < 0.06)
                
                symptom_dur = int(np.clip(np.random.normal(190, 110), 14, 600))
                symptom_onset = np.random.choice(['Gradual', 'Sudden'], p=[0.92, 0.08])
                symptom_sev = np.choice = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.50, 0.44, 0.06])
                symptom_sev = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.50, 0.44, 0.06])
                affected_eye = np.random.choice(['Both', 'Left', 'Right'], p=[0.82, 0.09, 0.09])
                
                iop_base = np.random.normal(15.3, 2.4)
                iop_l = round(float(np.clip(iop_base + np.random.normal(0, 0.6), 10.5, 21.0)), 1)
                iop_r = round(float(np.clip(iop_base + np.random.normal(0, 0.6), 10.5, 21.0)), 1)
                
                va_base = np.random.normal(0.44, 0.18)
                va_l = round(float(np.clip(va_base + np.random.normal(0, 0.08), 0.12, 0.95)), 2)
                va_r = round(float(np.clip(va_base + np.random.normal(0, 0.08), 0.12, 0.95)), 2)
                
                cdr_base = np.random.normal(0.31, 0.07)
                cdr_l = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.16, 0.48)), 2)
                cdr_r = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.16, 0.48)), 2)
                
                fam_glauc = int(np.random.rand() < 0.08)
                prev_glauc = int(np.random.rand() < 0.01)
                
                cdr_missing = (np.random.rand() < 0.52)
                iop_missing = (np.random.rand() < 0.05)

            elif cls == 'Dry_Eye_Disease':
                age = int(np.clip(np.random.normal(47, 14), 20, 78))
                gender = np.random.choice(['Male', 'Female'], p=[0.38, 0.62])
                occ_p = [0.30, 0.06, 0.06, 0.10, 0.10, 0.04, 0.14, 0.08, 0.06, 0.06]
                diabetes = int(np.random.rand() < 0.15)
                hypertension = int(np.random.rand() < 0.25)
                fam_eye = int(np.random.rand() < 0.22)
                prev_surg = int(np.random.rand() < 0.16)
                prev_inj = int(np.random.rand() < 0.04)
                contact_lens = int(np.random.rand() < 0.42)
                smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.65, 0.18, 0.17])
                
                blurred_vis = int(np.random.rand() < 0.46)
                floaters = int(np.random.rand() < 0.07)
                eye_pain = int(np.random.rand() < 0.40)
                eye_redness = int(np.random.rand() < 0.60)
                eye_itch = int(np.random.rand() < 0.52)
                eye_dry = int(np.random.rand() < 0.94)
                watering = int(np.random.rand() < 0.58)
                diplopia = int(np.random.rand() < 0.05)
                flashes = int(np.random.rand() < 0.02)
                photophobia = int(np.random.rand() < 0.48)
                night_blind = int(np.random.rand() < 0.12)
                swelling = int(np.random.rand() < 0.14)
                headache = int(np.random.rand() < 0.22)
                discharge = int(np.random.rand() < 0.16)
                foreign_body = int(np.random.rand() < 0.82)
                
                symptom_dur = int(np.clip(np.random.normal(160, 95), 14, 500))
                symptom_onset = np.random.choice(['Gradual', 'Sudden'], p=[0.82, 0.18])
                symptom_sev = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.40, 0.48, 0.12])
                affected_eye = np.random.choice(['Both', 'Left', 'Right'], p=[0.88, 0.06, 0.06])
                
                iop_base = np.random.normal(15.0, 2.3)
                iop_l = round(float(np.clip(iop_base + np.random.normal(0, 0.6), 10.0, 20.5)), 1)
                iop_r = round(float(np.clip(iop_base + np.random.normal(0, 0.6), 10.0, 20.5)), 1)
                
                va_base = np.random.normal(0.08, 0.06)
                va_l = round(float(np.clip(va_base + np.random.normal(0, 0.04), 0.00, 0.28)), 2)
                va_r = round(float(np.clip(va_base + np.random.normal(0, 0.04), 0.00, 0.28)), 2)
                
                cdr_base = np.random.normal(0.30, 0.06)
                cdr_l = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.16, 0.46)), 2)
                cdr_r = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.16, 0.46)), 2)
                
                fam_glauc = int(np.random.rand() < 0.07)
                prev_glauc = int(np.random.rand() < 0.02)
                
                cdr_missing = (np.random.rand() < 0.62)
                iop_missing = (np.random.rand() < 0.06)

            elif cls == 'Glaucoma':
                age = int(np.clip(np.random.normal(63, 9), 40, 86))
                gender = np.random.choice(['Male', 'Female'], p=[0.49, 0.51])
                occ_p = [0.10, 0.10, 0.08, 0.06, 0.08, 0.12, 0.32, 0.00, 0.08, 0.06]
                diabetes = int(np.random.rand() < 0.20)
                hypertension = int(np.random.rand() < 0.46)
                fam_eye = int(np.random.rand() < 0.50)
                prev_surg = int(np.random.rand() < 0.14)
                prev_inj = int(np.random.rand() < 0.09)
                contact_lens = int(np.random.rand() < 0.05)
                smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.55, 0.27, 0.18])
                
                is_asymptomatic = (np.random.rand() < 0.30)
                
                blurred_vis = int(not is_asymptomatic and np.random.rand() < 0.55)
                floaters = int(np.random.rand() < 0.16)
                eye_pain = int(not is_asymptomatic and np.random.rand() < 0.26)
                eye_redness = int(not is_asymptomatic and np.random.rand() < 0.18)
                eye_itch = int(np.random.rand() < 0.06)
                eye_dry = int(np.random.rand() < 0.24)
                watering = int(np.random.rand() < 0.14)
                diplopia = int(np.random.rand() < 0.05)
                flashes = int(np.random.rand() < 0.05)
                photophobia = int(not is_asymptomatic and np.random.rand() < 0.32)
                night_blind = int(not is_asymptomatic and np.random.rand() < 0.52)
                swelling = 0
                headache = int(not is_asymptomatic and np.random.rand() < 0.36)
                discharge = 0
                foreign_body = int(np.random.rand() < 0.08)
                
                if is_asymptomatic:
                    symptom_dur = 0
                    symptom_onset = 'None'
                    symptom_sev = 'None'
                    affected_eye = 'None'
                else:
                    symptom_dur = int(np.clip(np.random.normal(260, 140), 14, 730))
                    symptom_onset = np.random.choice(['Gradual', 'Sudden'], p=[0.82, 0.18])
                    symptom_sev = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.48, 0.36, 0.16])
                    affected_eye = np.random.choice(['Both', 'Left', 'Right'], p=[0.74, 0.13, 0.13])
                
                is_ntg = (np.random.rand() < 0.26)
                if is_ntg:
                    iop_base = np.random.normal(17.2, 2.2)
                else:
                    iop_base = np.random.normal(25.4, 4.4)
                    
                iop_l = round(float(np.clip(iop_base + np.random.normal(0, 1.2), 12.5, 42.0)), 1)
                iop_r = round(float(np.clip(iop_base + np.random.normal(0, 1.2), 12.5, 42.0)), 1)
                
                va_base = np.random.normal(0.24, 0.16)
                va_l = round(float(np.clip(va_base + np.random.normal(0, 0.08), 0.02, 0.85)), 2)
                va_r = round(float(np.clip(va_base + np.random.normal(0, 0.08), 0.02, 0.85)), 2)
                
                cdr_base = np.random.normal(0.64, 0.11)
                asym = np.random.normal(0, 0.05)
                cdr_l = round(float(np.clip(cdr_base + asym, 0.40, 0.88)), 2)
                cdr_r = round(float(np.clip(cdr_base - asym, 0.40, 0.88)), 2)
                
                fam_glauc = int(np.random.rand() < 0.42)
                prev_glauc = int(np.random.rand() < 0.22)
                
                cdr_missing = (np.random.rand() < 0.06)
                iop_missing = (np.random.rand() < 0.03)

            elif cls == 'Conjunctivitis':
                age = int(np.clip(np.random.normal(32, 12), 18, 70))
                gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
                occ_p = [0.18, 0.14, 0.08, 0.14, 0.14, 0.06, 0.04, 0.10, 0.06, 0.06]
                diabetes = int(np.random.rand() < 0.07)
                hypertension = int(np.random.rand() < 0.14)
                fam_eye = int(np.random.rand() < 0.12)
                prev_surg = int(np.random.rand() < 0.04)
                prev_inj = int(np.random.rand() < 0.05)
                contact_lens = int(np.random.rand() < 0.32)
                smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.68, 0.18, 0.14])
                
                blurred_vis = int(np.random.rand() < 0.22)
                floaters = int(np.random.rand() < 0.03)
                eye_pain = int(np.random.rand() < 0.54)
                eye_redness = int(np.random.rand() < 0.94)
                eye_itch = int(np.random.rand() < 0.74)
                eye_dry = int(np.random.rand() < 0.20)
                watering = int(np.random.rand() < 0.76)
                diplopia = 0
                flashes = 0
                photophobia = int(np.random.rand() < 0.44)
                night_blind = 0
                swelling = int(np.random.rand() < 0.46)
                headache = int(np.random.rand() < 0.14)
                discharge = int(np.random.rand() < 0.86)
                foreign_body = int(np.random.rand() < 0.70)
                
                symptom_dur = int(np.clip(np.random.exponential(5) + 1, 1, 21))
                symptom_onset = np.random.choice(['Sudden', 'Gradual'], p=[0.84, 0.16])
                symptom_sev = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.28, 0.56, 0.16])
                affected_eye = np.random.choice(['Both', 'Left', 'Right'], p=[0.52, 0.24, 0.24])
                
                iop_base = np.random.normal(15.1, 2.3)
                iop_l = round(float(np.clip(iop_base + np.random.normal(0, 0.6), 10.0, 20.5)), 1)
                iop_r = round(float(np.clip(iop_base + np.random.normal(0, 0.6), 10.0, 20.5)), 1)
                
                va_base = np.random.normal(0.04, 0.04)
                va_l = round(float(np.clip(va_base + np.random.normal(0, 0.03), 0.00, 0.18)), 2)
                va_r = round(float(np.clip(va_base + np.random.normal(0, 0.03), 0.00, 0.18)), 2)
                
                cdr_base = np.random.normal(0.29, 0.06)
                cdr_l = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.15, 0.45)), 2)
                cdr_r = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.15, 0.45)), 2)
                
                fam_glauc = int(np.random.rand() < 0.05)
                prev_glauc = int(np.random.rand() < 0.01)
                
                cdr_missing = (np.random.rand() < 0.78)
                iop_missing = (np.random.rand() < 0.42)

            elif cls == 'Diabetic_Retinopathy':
                age = int(np.clip(np.random.normal(58, 9), 34, 82))
                gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])
                occ_p = [0.12, 0.12, 0.08, 0.06, 0.08, 0.14, 0.26, 0.00, 0.08, 0.06]
                diabetes = int(np.random.rand() < 0.95)
                hypertension = int(np.random.rand() < 0.68)
                fam_eye = int(np.random.rand() < 0.32)
                prev_surg = int(np.random.rand() < 0.15)
                prev_inj = int(np.random.rand() < 0.04)
                contact_lens = int(np.random.rand() < 0.04)
                smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.50, 0.32, 0.18])
                
                is_asymptomatic = (np.random.rand() < 0.25)
                
                blurred_vis = int(not is_asymptomatic and np.random.rand() < 0.86)
                floaters = int(not is_asymptomatic and np.random.rand() < 0.62)
                eye_pain = int(np.random.rand() < 0.06)
                eye_redness = int(np.random.rand() < 0.06)
                eye_itch = int(np.random.rand() < 0.04)
                eye_dry = int(np.random.rand() < 0.18)
                watering = int(np.random.rand() < 0.09)
                diplopia = int(np.random.rand() < 0.12)
                flashes = int(not is_asymptomatic and np.random.rand() < 0.30)
                photophobia = int(np.random.rand() < 0.20)
                night_blind = int(not is_asymptomatic and np.random.rand() < 0.44)
                swelling = int(np.random.rand() < 0.03)
                headache = int(np.random.rand() < 0.15)
                discharge = 0
                foreign_body = int(np.random.rand() < 0.05)
                
                if is_asymptomatic:
                    symptom_dur = 0
                    symptom_onset = 'None'
                    symptom_sev = 'None'
                    affected_eye = 'None'
                else:
                    symptom_dur = int(np.clip(np.random.normal(230, 130), 20, 650))
                    symptom_onset = np.random.choice(['Gradual', 'Sudden'], p=[0.76, 0.24])
                    symptom_sev = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.38, 0.42, 0.20])
                    affected_eye = np.random.choice(['Both', 'Left', 'Right'], p=[0.80, 0.10, 0.10])
                
                iop_base = np.random.normal(16.3, 2.8)
                iop_l = round(float(np.clip(iop_base + np.random.normal(0, 0.7), 11.0, 24.5)), 1)
                iop_r = round(float(np.clip(iop_base + np.random.normal(0, 0.7), 11.0, 24.5)), 1)
                
                va_base = np.random.normal(0.42, 0.18)
                va_l = round(float(np.clip(va_base + np.random.normal(0, 0.08), 0.10, 1.05)), 2)
                va_r = round(float(np.clip(va_base + np.random.normal(0, 0.08), 0.10, 1.05)), 2)
                
                cdr_base = np.random.normal(0.34, 0.08)
                cdr_l = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.18, 0.52)), 2)
                cdr_r = round(float(np.clip(cdr_base + np.random.normal(0, 0.02), 0.18, 0.52)), 2)
                
                fam_glauc = int(np.random.rand() < 0.09)
                prev_glauc = int(np.random.rand() < 0.02)
                
                cdr_missing = (np.random.rand() < 0.14)
                iop_missing = (np.random.rand() < 0.04)

            occ = np.random.choice(occupations, p=occ_p)
            
            final_iop_l = np.nan if iop_missing else iop_l
            final_iop_r = np.nan if iop_missing else iop_r
            final_cdr_l = np.nan if cdr_missing else cdr_l
            final_cdr_r = np.nan if cdr_missing else cdr_r
            
            rec = {
                'patient_id': p_id,
                'age': age,
                'gender': gender,
                'occupation': occ,
                'diabetes_history': diabetes,
                'hypertension_history': hypertension,
                'family_history_eye_disease': fam_eye,
                'previous_eye_surgery': prev_surg,
                'previous_eye_injury': prev_inj,
                'contact_lens_usage': contact_lens,
                'smoking_history': smoking,
                
                'blurred_vision': blurred_vis,
                'black_spots_floaters': floaters,
                'eye_pain': eye_pain,
                'eye_redness': eye_redness,
                'eye_itching': eye_itch,
                'eye_dryness': eye_dry,
                'excessive_watering': watering,
                'double_vision': diplopia,
                'flashes_of_light': flashes,
                'light_sensitivity': photophobia,
                'difficulty_seeing_at_night': night_blind,
                'eye_swelling': swelling,
                'headache': headache,
                'discharge_from_eye': discharge,
                'foreign_body_sensation': foreign_body,
                'symptom_duration_days': symptom_dur,
                'symptom_onset': symptom_onset,
                'symptom_severity': symptom_sev,
                'affected_eye': affected_eye,
                
                'intraocular_pressure_left': final_iop_l,
                'intraocular_pressure_right': final_iop_r,
                'visual_acuity_left': va_l,
                'visual_acuity_right': va_r,
                'cup_to_disc_ratio_left': final_cdr_l,
                'cup_to_disc_ratio_right': final_cdr_r,
                'family_history_glaucoma': fam_glauc,
                'previous_glaucoma_diagnosis': prev_glauc,
                
                'eye_disease': cls
            }
            records.append(rec)
            
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df['patient_id'] = [f"PID-{i+1:05d}" for i in range(len(df))]
    
    return df

if __name__ == '__main__':
    df = generate_eye_disease_dataset(n_samples=5000, random_seed=42)
    csv_path = 'eye_disease_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records to {csv_path}")
    print("Class distribution:\n", df['eye_disease'].value_counts())
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
