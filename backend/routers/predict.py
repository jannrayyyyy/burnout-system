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
    # Validate required fields before prediction
    missing = [f for f in REQUIRED if f not in payload.data]
    if missing:
        raise HTTPException(status_code=400, detail={"missing": missing})

    result = run_prediction(payload.data)

    # Save to canonical survey_responses collection. run_prediction already logs an
    # entry into `untrained_surveys` with prediction metadata, so avoid duplicating that.
    try:
        db.collection("survey_responses").add({
            "data": payload.data,
            "prediction": result.get("burnout_level"),
            "probability": result.get("probability"),
            "created_at": datetime.utcnow().isoformat(),
        })
    except Exception:
        # don't fail the API if saving to Firestore fails; warn instead
        raise HTTPException(status_code=500, detail="Failed to save survey to database")

    return {"message": "Survey submitted", **result}
