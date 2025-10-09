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
    """Submit a survey, run prediction, and save to Firebase (single entry)."""
    missing = [f for f in REQUIRED if f not in payload.data]
    if missing:
        raise HTTPException(status_code=400, detail={"missing": missing})

    # run_prediction already logs prediction into Firestore
    try:
        result = run_prediction(payload.data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {"message": "Survey submitted", **result}
