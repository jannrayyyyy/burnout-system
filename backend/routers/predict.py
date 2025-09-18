from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
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
    result = run_prediction(payload.data)

    doc_ref = db.collection("survey_responses").document()
    doc_ref.set({
        "created_at": datetime.utcnow().isoformat(),
        "data": payload.data,
        "prediction": result["burnout_level"],
        "probability": result["probability"]
    })

    return {"id": doc_ref.id, **result}
