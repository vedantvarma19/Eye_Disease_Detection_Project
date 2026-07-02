# src/recommendation.py

def get_clinical_guidance(class_id):
    """
    Acts as an analytical backend engine, mapping model integer class outputs
    to explicit, readable clinical diagnoses and treatment precautions.
    """
    # Defensive lookup dictionary mapping class IDs to structured clinical details
    disease_database = {
        0: {
            "name": "CNV (Choroidal Neovascularization)",
            "description": "Abnormal growth of new, fragile blood vessels beneath the retina, often leading to fluid leakage and vision distortion.",
            "urgency": "HIGH - Immediate Ophthalmologist Consultation Required",
            "precautions": [
                "Schedule a high-priority dilated eye exam immediately.",
                "Monitor vision daily using an Amsler grid chart.",
                "Avoid strenuous heavy weightlifting or high-impact activities that increase intraocular pressure."
            ]
        },
        1: {
            "name": "DME (Diabetic Macular Edema)",
            "description": "An accumulation of fluid in the macula caused by leaking blood vessels, directly related to underlying complications from diabetes.",
            "urgency": "MEDIUM TO HIGH - Targeted Clinical Intervention Needed",
            "precautions": [
                "Consult both an endocrinologist and an ophthalmologist.",
                "Strictly track and manage blood glucose levels, HbA1c, and blood pressure.",
                "Maintain a low-glycemic, heart-healthy diet."
            ]
        },
        2: {
            "name": "DRUSEN (Early Age-Related Macular Degeneration)",
            "description": "Small yellow or white deposits of fatty proteins that accumulate beneath the retina, indicating early-stage dry macular degeneration.",
            "urgency": "MEDIUM - Routine Monitoring and Lifestyle Maintenance",
            "precautions": [
                "Schedule comprehensive eye examinations every 6 to 12 months.",
                "Incorporate antioxidant-rich foods (leafy greens, zinc) or specialized eye vitamins.",
                "Quit smoking immediately, as tobacco usage accelerates macular degradation."
            ]
        },
        3: {
            "name": "NORMAL",
            "description": "The retinal layers appear perfectly intact, uniform, healthy, and free of visible fluid pockets, accumulations, or structural lesions.",
            "urgency": "ROUTINE - Standard Preventive Maintenance",
            "precautions": [
                "Continue scheduling standard annual comprehensive preventive eye checkups.",
                "Protect eyes from excess UV exposure using high-quality sunglasses.",
                "Follow the 20-20-20 rule when working on digital screens to reduce fatigue."
            ]
        }
    }
    
    # Return the matching dictionary block, or fall back to a safe default if an out-of-bounds index is passed
    return disease_database.get(class_id, {
        "name": "Unknown Assessment Profile",
        "description": "The target diagnostic ID is out of bounds or cannot be parsed by the system database entries.",
        "urgency": "UNKNOWN",
        "precautions": ["Consult a medical practitioner immediately."]
    })

if __name__ == "__main__":
    print("========= CLINICAL RECOMMENDATION ENGINE TEST =========")
    
    # Simulate a scenario where our CNN predicted class 1 (DME) with high confidence
    mock_predicted_id = 1
    mock_confidence_score = 0.9421
    
    # Query our rules engine dictionary database
    guidance = get_clinical_guidance(mock_predicted_id)
    
    print(f"[INPUT] Model Classified ID   : {mock_predicted_id}")
    print(f"[INPUT] Confidence Score      : {mock_confidence_score * 100:.2f}%")
    print("------------------------------------------------------")
    print(f"📋 DIAGNOSIS       : {guidance['name']}")
    print(f"📖 PATHOLOGY       : {guidance['description']}")
    print(f"🚨 URGENCY PROFILE  : {guidance['urgency']}")
    print("⚡ IMMEDIATE ACTION PRECAUTIONS:")
    for idx, precaution in enumerate(guidance['precautions'], 1):
        print(f"   {idx}. {precaution}")
    print("=======================================================\n")