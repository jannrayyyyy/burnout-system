from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
from backend.services.model_service import run_prediction
from backend.services.firebase_service import db

router = APIRouter()

class InputPayload(BaseModel):
    data: Dict[str, Any]

REQUIRED = [
    "age", "gender", "year_level", "gwa", "num_subjects",
    "hours_online", "study_hours", "sleep_hours",
    "perceived_stress", "procrastination", "motivation"
]

@router.post("/predict")
def predict(payload: InputPayload):
    data = payload.data
    missing = [f for f in REQUIRED if f not in data]
    if missing:
        raise HTTPException(status_code=400, detail={"missing": missing})

    return run_prediction(data)

@router.post("/submit")
def submit(payload: InputPayload):
    """Submit a survey, run prediction, and save to Firebase (both survey_responses + untrained_surveys)."""
    result = run_prediction(payload.data)
    doc_data = {
        "data": payload.data,
        "all_probabilities": result["all_probabilities"],
        "probability": result["probability"],
        "burnout_level":result["burnout_level"],
    }

    # Save to main collection
    db.collection("survey_responses").add(doc_data)
    # Also add to untrained_surveys
    db.collection("untrained_surveys").add(doc_data)

    return {"message": "Survey submitted", **result}
