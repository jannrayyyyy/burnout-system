from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from backend.services.prediction_service import run_prediction
from backend.services.firebase_service import db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# -----------------------------------
# Models
# -----------------------------------

class InputPayload(BaseModel):
    data: Dict[str, Any] = Field(..., description="Survey data for burnout prediction")

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError("data must be a non-empty dictionary")
        return v


REQUIRED_FIELDS = [
    "gender",
    "year_level",
    "gwa",
    "how_far_is_your_home_from_school_(one_way)",
    "what_type_of_learning_modality_do_you_currently_attend",
    "sleep_patterns_and_physical_health_i_usually_get_less_than_6_hours_of_sleep_on_school_nights",
    "sleep_patterns_and_physical_health_i_find_it_difficult_to_fall_asleep_because_of_academic_stress",
    "sleep_patterns_and_physical_health_i_often_wake_up_feeling_tired_or_unrefreshed",
    "academic_workload_and_study_habits_i_find_my_academic_workload_unmanageable",
    "emotional_state_and_burnout_indicators_i_feel_emotionally_drained_at_the_end_of_the_school_day",
    "home_environment_and_personal_stress_i_feel_that_my_personal_or_home_life_affects_my_academic_performance",
    "motivation_and_personal_accomplishment_i_feel_like_i_am_not_accomplishing_anything_worthwhile_in_school",
]

# -----------------------------------
# Routes
# -----------------------------------

@router.post("/", summary="Predict burnout risk", status_code=status.HTTP_200_OK)
async def predict(payload: InputPayload):
    """
    Run burnout prediction using the trained ML model.
    """
    data = payload.data

    # Validate required fields
    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in (None, "", " ")]
    if missing:
        logger.warning(f"Missing fields in request: {missing}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Missing required fields", "missing_fields": missing},
        )

    try:
        prediction = run_prediction(data)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction process failed: {str(e)}",
        )

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": prediction,
    }


@router.post("/submit", summary="Submit survey and run burnout prediction", status_code=status.HTTP_201_CREATED)
async def submit(payload: InputPayload):
    """
    Submits survey data to Firebase and runs burnout prediction.
    """
    data = payload.data

    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in (None, "", " ")]
    if missing:
        logger.warning(f"Missing fields in submission: {missing}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Missing required fields", "missing_fields": missing},
        )

    try:
        result = run_prediction(data)
    except Exception as e:
        logger.exception("Prediction execution failed during submission")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

    return {
        "message": "Survey submitted successfully",
        "timestamp": datetime.utcnow().isoformat(),
        "result": result,
    }

# -----------------------------------
# Health check endpoint (for production readiness)
# -----------------------------------

@router.get("/health", summary="Service health check")
async def health_check():
    try:
        if db is not None:
            _ = db.collection("surveys").limit(1).get()
        return {"status": "ok", "firebase": bool(db is not None)}
    except Exception:
        return {"status": "degraded", "firebase": False}
