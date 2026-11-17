import pandas as pd

data_dict = [
    {
        "column_name": "patient_id",
        "data_type": "string",
        "description": "Unique synthetic patient alphanumeric identification code",
        "allowed_values": "PID-00001 to PID-05000",
        "units": "None"
    },
    {
        "column_name": "age",
        "data_type": "integer",
        "description": "Patient biological age in completed chronological years at examination",
        "allowed_values": "18 to 88",
        "units": "Years"
    },
    {
        "column_name": "gender",
        "data_type": "categorical (string)",
        "description": "Biological sex / self-reported gender of the patient",
        "allowed_values": "Male, Female",
        "units": "None"
    },
    {
        "column_name": "occupation",
        "data_type": "categorical (string)",
        "description": "Primary vocational category / occupational environment",
        "allowed_values": "Office / Desk Worker, Manual Laborer / Construction, Driver / Transport, Healthcare Professional, Teacher / Academic, Farmer / Agricultural, Retired / Elderly, Student, Homemaker, Sales / Retail",
        "units": "None"
    },
    {
        "column_name": "diabetes_history",
        "data_type": "binary (integer)",
        "description": "Clinical diagnosis or self-reported history of diabetes mellitus (Type 1 or 2)",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "hypertension_history",
        "data_type": "binary (integer)",
        "description": "Documented medical history of systemic arterial hypertension",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "family_history_eye_disease",
        "data_type": "binary (integer)",
        "description": "Presence of major ocular disease in first-degree biological relatives (e.g., glaucoma, cataract, retinal dystrophy, severe myopia)",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "previous_eye_surgery",
        "data_type": "binary (integer)",
        "description": "Surgical history involving either eye (e.g., cataract extraction, LASIK/PRK, trabeculectomy, retinal laser photocoagulation)",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "previous_eye_injury",
        "data_type": "binary (integer)",
        "description": "History of mechanical, blunt, sharp, or chemical trauma to ocular tissues",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "contact_lens_usage",
        "data_type": "binary (integer)",
        "description": "Current active or regular use of soft, rigid gas permeable, or cosmetic contact lenses",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "smoking_history",
        "data_type": "categorical (string)",
        "description": "Lifetime tobacco cigarette smoking classification status",
        "allowed_values": "Never, Former, Current",
        "units": "None"
    },
    {
        "column_name": "blurred_vision",
        "data_type": "binary (integer)",
        "description": "Subjective complaint of decreased visual clarity, fuzziness, or loss of sharpness at near or distance",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "black_spots_floaters",
        "data_type": "binary (integer)",
        "description": "Perception of moving dark specks, cobwebs, strings, or spots drifting across the visual field (vitreous floaters/hemorrhage)",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "eye_pain",
        "data_type": "binary (integer)",
        "description": "Ocular discomfort, deep ache, retrobulbar pain, or sharp stinging sensation",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "eye_redness",
        "data_type": "binary (integer)",
        "description": "Hyperemia of conjunctival, episcleral, or ciliary vasculature ('pink/red eye')",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "eye_itching",
        "data_type": "binary (integer)",
        "description": "Pruritus or tickling sensation of the ocular surface, canthi, or lid margins",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "eye_dryness",
        "data_type": "binary (integer)",
        "description": "Sensation of insufficient tear lubrication, burning, scratchiness, or desiccated ocular surface",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "excessive_watering",
        "data_type": "binary (integer)",
        "description": "Epiphora or hyperlacrimation caused by reflex irritation or lacrimal drainage impairment",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "double_vision",
        "data_type": "binary (integer)",
        "description": "Diplopia: seeing two overlapping or separated images of a single object (monocular or binocular)",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "flashes_of_light",
        "data_type": "binary (integer)",
        "description": "Photopsia: perceived streaks, lightning arcs, or flashes of light in the peripheral visual field",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "light_sensitivity",
        "data_type": "binary (integer)",
        "description": "Photophobia or glare sensitivity provoked by normal indoor or outdoor light levels",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "difficulty_seeing_at_night",
        "data_type": "binary (integer)",
        "description": "Nyctalopia: impaired visual contrast, glare, or subnormal dark adaptation in mesopic or scotopic illumination",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "eye_swelling",
        "data_type": "binary (integer)",
        "description": "Edema of periorbital tissues, eyelids, or conjunctival chemosis",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "headache",
        "data_type": "binary (integer)",
        "description": "Cephalea, brow ache, temple pressure, or frontal asthenopic headache associated with eye strain or IOP changes",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "discharge_from_eye",
        "data_type": "binary (integer)",
        "description": "Mucopurulent, purulent, serous, or crusting exudate accumulating on eyelids, lashes, or canthi",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "foreign_body_sensation",
        "data_type": "binary (integer)",
        "description": "Feeling of sand, grit, glass, or an eyelash lodged under the eyelid or on the corneal surface",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "symptom_duration_days",
        "data_type": "integer",
        "description": "Estimated elapsed duration in days from symptom onset to current clinical triage (0 if asymptomatic)",
        "allowed_values": "0 to 730",
        "units": "Days"
    },
    {
        "column_name": "symptom_onset",
        "data_type": "categorical (string)",
        "description": "Temporal nature and speed of symptom manifestation",
        "allowed_values": "Sudden, Gradual, None",
        "units": "None"
    },
    {
        "column_name": "symptom_severity",
        "data_type": "categorical (string)",
        "description": "Patient-reported subjective overall intensity of presenting ocular symptoms",
        "allowed_values": "Mild, Moderate, Severe, None",
        "units": "None"
    },
    {
        "column_name": "affected_eye",
        "data_type": "categorical (string)",
        "description": "Anatomical laterality of the presenting ocular complaints or signs",
        "allowed_values": "Left, Right, Both, None",
        "units": "None"
    },
    {
        "column_name": "intraocular_pressure_left",
        "data_type": "float",
        "description": "Intraocular pressure (IOP) of the left eye measured via tonometry (Goldmann or non-contact); missing if deferred/omitted",
        "allowed_values": "10.0 to 42.0 (or NaN/missing)",
        "units": "mmHg"
    },
    {
        "column_name": "intraocular_pressure_right",
        "data_type": "float",
        "description": "Intraocular pressure (IOP) of the right eye measured via tonometry (Goldmann or non-contact); missing if deferred/omitted",
        "allowed_values": "10.0 to 42.0 (or NaN/missing)",
        "units": "mmHg"
    },
    {
        "column_name": "visual_acuity_left",
        "data_type": "float",
        "description": "Best-measured or presenting visual acuity of the left eye expressed on the standardized LogMAR continuous scale (0.00 = 20/20 standard normal; 1.00 = 20/200; higher values indicate worse acuity)",
        "allowed_values": "0.00 to 1.20",
        "units": "LogMAR"
    },
    {
        "column_name": "visual_acuity_right",
        "data_type": "float",
        "description": "Best-measured or presenting visual acuity of the right eye expressed on the standardized LogMAR continuous scale (0.00 = 20/20 standard normal; 1.00 = 20/200; higher values indicate worse acuity)",
        "allowed_values": "0.00 to 1.20",
        "units": "LogMAR"
    },
    {
        "column_name": "cup_to_disc_ratio_left",
        "data_type": "float",
        "description": "Vertical optic nerve head cup-to-disc ratio (CDR) of the left eye evaluated via fundoscopy / slit lamp biomicroscopy; missing if dilated exam not indicated/performed",
        "allowed_values": "0.15 to 0.90 (or NaN/missing)",
        "units": "Ratio (dimensionless 0 to 1)"
    },
    {
        "column_name": "cup_to_disc_ratio_right",
        "data_type": "float",
        "description": "Vertical optic nerve head cup-to-disc ratio (CDR) of the right eye evaluated via fundoscopy / slit lamp biomicroscopy; missing if dilated exam not indicated/performed",
        "allowed_values": "0.15 to 0.90 (or NaN/missing)",
        "units": "Ratio (dimensionless 0 to 1)"
    },
    {
        "column_name": "family_history_glaucoma",
        "data_type": "binary (integer)",
        "description": "Specific positive family history of open-angle or angle-closure glaucoma in direct biological relatives",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "previous_glaucoma_diagnosis",
        "data_type": "binary (integer)",
        "description": "Prior documented medical diagnosis of glaucoma or ongoing prescription for ocular hypotensive drops",
        "allowed_values": "0 = No, 1 = Yes",
        "units": "None"
    },
    {
        "column_name": "eye_disease",
        "data_type": "categorical (string)",
        "description": "Target clinical diagnostic classification assigned based on comprehensive multi-feature synthetic assessment",
        "allowed_values": "Normal, Glaucoma, Cataract, Diabetic_Retinopathy, Conjunctivitis, Dry_Eye_Disease, Refractive_Error",
        "units": "None"
    }
]

df_dict = pd.DataFrame(data_dict)
df_dict.to_csv("data_dictionary.csv", index=False)
print("Saved data_dictionary.csv with", len(df_dict), "entries.")
