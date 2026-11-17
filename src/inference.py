"""
Production Inference and Clinical Decision Support Module
Project: AI-Based Eye Disease Prediction and Clinical Decision Support System
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.data_preprocessing import ALL_PREDICTIVE_FEATURES


class EyeDiseasePredictor:
    """Production predictor for the Myopsia Clinical Decision Support System."""
    
    def __init__(
        self,
        model_path: str = 'saved_models/eye_disease_pipeline.joblib',
        metadata_path: str = 'saved_models/model_metadata.json'
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        self.pipeline = joblib.load(model_path)
        self.classes_ = list(self.pipeline.classes_)
        
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
    def _format_input(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates and formats raw patient dictionary into a 1-row DataFrame.
        Missing fields are populated with NaN so the pipeline imputer can handle them.
        """
        row = {}
        for feature in ALL_PREDICTIVE_FEATURES:
            val = raw_data.get(feature, np.nan)
            if val == "" or val is None:
                val = np.nan
            row[feature] = val
            
        return pd.DataFrame([row])
        
    def _get_clinical_recommendations(self, predicted_class: str, confidence: float, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates clinical triage urgency and next-step diagnostic recommendations."""
        recommendations = {
            'Normal': {
                'triage_level': 'Routine / Wellness',
                'urgency': 'Low',
                'summary': 'Clinical parameters and symptom profile conform to physiological norms.',
                'next_steps': [
                    'Schedule standard annual or biannual routine vision checkup.',
                    'Advise ergonomic screen distance and 20-20-20 rule for digital eyestrain.',
                    'Seek prompt re-evaluation if new sudden visual disturbances emerge.'
                ]
            },
            'Glaucoma': {
                'triage_level': 'Urgent Ophthalmic Referral',
                'urgency': 'High',
                'summary': 'Optic nerve cupping and/or elevated intraocular pressure suggest glaucomatous neuropathy.',
                'next_steps': [
                    'Urgent referral to an ophthalmologist / glaucoma specialist.',
                    'Order 24-2 or 30-2 Humphrey Visual Field perimetry.',
                    'Perform gonioscopy to evaluate anterior chamber angle closure vs. open angle.',
                    'Obtain retinal nerve fiber layer (RNFL) OCT scan.'
                ]
            },
            'Cataract': {
                'triage_level': 'Ophthalmic Outpatient Consultation',
                'urgency': 'Moderate',
                'summary': 'Age profile and gradual loss of visual clarity indicate senile crystalline lens opacification.',
                'next_steps': [
                    'Slit-lamp biomicroscopy after pharmacological mydriasis.',
                    'Assess functional impact on activities of daily living and driving.',
                    'Plan for biometry and phacoemulsification with intraocular lens (IOL) implantation when indicated.'
                ]
            },
            'Diabetic_Retinopathy': {
                'triage_level': 'Priority Retinal Triage',
                'urgency': 'High',
                'summary': 'Retinal microvascular pathology indicated by diabetes history, floaters, and acuity changes.',
                'next_steps': [
                    'Immediate dilated fundus examination by a vitreoretinal specialist.',
                    'Order optical coherence tomography (OCT) for diabetic macular edema (DME).',
                    'Coordinate with endocrinology/primary physician for glycemic (HbA1c) and blood pressure optimization.',
                    'Assess eligibility for anti-VEGF intravitreal therapy or panretinal photocoagulation.'
                ]
            },
            'Conjunctivitis': {
                'triage_level': 'Primary Care / Acute Triage',
                'urgency': 'Moderate',
                'summary': 'Acute ocular surface inflammation presenting with hyperemia, discharge, and pruritus.',
                'next_steps': [
                    'Differentiate viral vs. bacterial vs. allergic etiology via slit lamp.',
                    'Instruct strict hand hygiene and separate towels to prevent contagious transmission.',
                    'Prescribe targeted topical broad-spectrum antibiotic or antihistamine drops.',
                    'Avoid contact lens wear until complete clinical resolution.'
                ]
            },
            'Dry_Eye_Disease': {
                'triage_level': 'Ocular Surface Management',
                'urgency': 'Low to Moderate',
                'summary': 'Tear film instability characterized by ocular surface grittiness, dryness, and reflex tearing.',
                'next_steps': [
                    'Perform Tear Break-Up Time (TBUT) and Schirmer test / corneal fluorescein staining.',
                    'Initiate preservative-free artificial tear lubricants and warm lid compresses.',
                    'Review contact lens wearing schedule and environmental humidity.',
                    'Consider topical cyclosporine or punctal occlusion for refractory cases.'
                ]
            },
            'Refractive_Error': {
                'triage_level': 'Optometric / Refraction Clinic',
                'urgency': 'Low',
                'summary': 'Subnormal uncorrected acuity and accommodative asthenopia responsive to optical correction.',
                'next_steps': [
                    'Perform comprehensive subjective and objective manifest refraction (dry and/or cycloplegic).',
                    'Prescribe corrective spectacle lenses or properly fitted contact lenses.',
                    'Evaluate for astigmatism or presbyopia depending on age.'
                ]
            }
        }
        return recommendations.get(predicted_class, {
            'triage_level': 'General Evaluation',
            'urgency': 'Moderate',
            'summary': 'Clinical assessment recommended.',
            'next_steps': ['Comprehensive ophthalmic examination indicated.']
        })

    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end inference for a single patient assessment.
        Returns predicted class, probabilities, triage level, and recommendations.
        """
        df_row = self._format_input(patient_data)
        
        # Predict class & probabilities
        pred_class = self.pipeline.predict(df_row)[0]
        pred_probs = self.pipeline.predict_proba(df_row)[0]
        
        prob_dict = {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(self.classes_, pred_probs)
        }
        # Sort probability distribution in descending order
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        
        confidence = float(np.max(pred_probs)) * 100
        clinical_guidance = self._get_clinical_recommendations(pred_class, confidence, patient_data)
        
        return {
            'predicted_condition': pred_class,
            'confidence_percentage': round(confidence, 2),
            'probability_distribution': sorted_probs,
            'clinical_guidance': clinical_guidance
        }


if __name__ == '__main__':
    predictor = EyeDiseasePredictor()
    sample_file = 'saved_models/sample_patient_payload.json'
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_patient = json.load(f)
        
    print("Testing Inference with Sample Patient:")
    result = predictor.predict(sample_patient)
    print(json.dumps(result, indent=2))
