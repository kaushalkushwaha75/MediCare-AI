import os
import json
import joblib
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="MediCare AI - Machine Learning Disease Prediction System",
    description="API for predicting probable diseases based on patient demographics & symptoms.",
    version="1.0.0"
)

# Enable CORS for VS Code Live Server and local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "symptom_encoder.joblib")
INFO_PATH = os.path.join(BASE_DIR, "model", "disease_info.json")

# Global variables for model artifacts
model = None
symptom_ids = []
disease_info_db = {}
symptom_catalog = []

def load_artifacts():
    global model, symptom_ids, disease_info_db, symptom_catalog
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            symptom_ids = joblib.load(ENCODER_PATH)
            print("Successfully loaded ML Model and Symptom Encoders.")
        else:
            print("Warning: ML Model artifacts not found yet. Run train_model.py first.")

        if os.path.exists(INFO_PATH):
            with open(INFO_PATH, "r") as f:
                data = json.load(f)
                disease_info_db = data.get("diseases", {})
                symptom_catalog = data.get("symptoms", [])
            print("Successfully loaded Disease Info knowledgebase.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

# Load artifacts on startup
load_artifacts()

# Pydantic Schemas
class PredictRequest(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    gender: str = Field("other", description="Gender: male, female, or other")
    duration_days: int = Field(3, ge=1, le=90, description="Symptom duration in days")
    severity_level: int = Field(5, ge=1, le=10, description="Self-reported severity scale 1-10")
    symptoms: List[str] = Field(..., min_length=1, description="List of active symptom IDs")

class SymptomItem(BaseModel):
    id: str
    name: str
    category: str

# Endpoints
@app.get("/api/symptoms")
def get_symptoms():
    """Returns the list of all available symptoms with category tags."""
    if not symptom_catalog:
        load_artifacts()
    return {"symptoms": symptom_catalog, "count": len(symptom_catalog)}

@app.post("/api/predict")
def predict_disease(payload: PredictRequest):
    """Predicts disease based on patient input vector using trained Random Forest ensemble."""
    global model, symptom_ids, disease_info_db

    if model is None or not symptom_ids:
        load_artifacts()
        if model is None:
            raise HTTPException(status_code=500, detail="ML Model not available. Please train model first.")

    # 1. Build binary feature vector with feature names matching model fitting
    import pandas as pd
    input_dict = {sym_id: [1 if sym_id in payload.symptoms else 0] for sym_id in symptom_ids}
    
    if sum(input_dict[sym_id][0] for sym_id in symptom_ids) == 0:
        raise HTTPException(status_code=400, detail="Please select at least one symptom.")

    # Convert to DataFrame with feature names
    X_input = pd.DataFrame(input_dict)

    # 2. Get class probabilities
    probabilities = model.predict_proba(X_input)[0]
    classes = model.classes_

    # Sort diseases by probability descending
    top_indices = np.argsort(probabilities)[::-1]
    
    predictions = []
    for idx in top_indices[:5]: # Top 5 differential diagnoses
        disease_name = classes[idx]
        prob = float(probabilities[idx])
        if prob > 0.01:
            predictions.append({
                "disease": disease_name,
                "confidence_percent": round(prob * 100, 1),
                "probability": prob
            })

    if not predictions:
        primary_disease = classes[top_indices[0]]
        predictions.append({
            "disease": primary_disease,
            "confidence_percent": round(float(probabilities[top_indices[0]]) * 100, 1),
            "probability": float(probabilities[top_indices[0]])
        })

    top_prediction = predictions[0]
    primary_name = top_prediction["disease"]

    # 3. Retrieve enriched medical details
    clinical_info = disease_info_db.get(primary_name, {
        "description": "Condition characterized by reported symptoms.",
        "severity": "Moderate" if payload.severity_level >= 5 else "Low",
        "specialist": "General Physician",
        "precautions": ["Rest adequately", "Stay well hydrated", "Consult a certified physician"],
        "emergency_warning": "High fever, difficulty breathing, or severe pain."
    })

    # Adjust severity assessment based on patient factors
    triage_level = clinical_info.get("severity", "Moderate")
    # Normalize compound values (e.g. 'Moderate to High' -> 'High', 'Low to Moderate' -> 'Moderate')
    if triage_level == "Moderate to High":
        triage_level = "High"
    elif triage_level == "Low to Moderate":
        triage_level = "Moderate"
    if payload.severity_level >= 8 or payload.age >= 65 or payload.duration_days > 14:
        if triage_level == "Moderate":
            triage_level = "High"
        elif triage_level == "Low":
            triage_level = "Moderate"

    response = {
        "status": "success",
        "patient_summary": {
            "age": payload.age,
            "gender": payload.gender.capitalize(),
            "duration_days": payload.duration_days,
            "severity_level": payload.severity_level,
            "symptom_count": len(payload.symptoms),
            "selected_symptoms": [s for s in symptom_catalog if s["id"] in payload.symptoms]
        },
        "primary_diagnosis": {
            "name": primary_name,
            "confidence": top_prediction["confidence_percent"],
            "triage_severity": triage_level,
            "description": clinical_info.get("description", ""),
            "recommended_specialist": clinical_info.get("specialist", "General Physician"),
            "precautions": clinical_info.get("precautions", []),
            "emergency_warning": clinical_info.get("emergency_warning", "")
        },
        "differential_diagnoses": predictions,
        "disclaimer": "MediCare AI is a predictive machine learning tool intended solely for informational purposes and triage assistance. It does not replace professional clinical evaluation, diagnosis, or treatment."
    }

    return response

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "symptoms_loaded": len(symptom_catalog) > 0,
        "disease_count": len(disease_info_db)
    }

@app.get("/")
def index():
    return {
        "service": "MediCare AI Backend API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "symptoms": "/api/symptoms",
            "predict": "/api/predict (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)