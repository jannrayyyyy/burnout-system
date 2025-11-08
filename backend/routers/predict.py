# backend/routers/predict.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import re

from backend.services.prediction_service import run_prediction, check_model_health
from backend.services.firebase_service import db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])


# ---------------------------------------------------------
# Field Mapping - Frontend to Model Features
# ---------------------------------------------------------
FIELD_MAPPING = {
    "academic_workload_unmanageable": "academic_workload_and_study_habits_i_find_my_academic_workload_unmanageable.",
    "confident_handling_challenges": "motivation_and_personal_accomplishment_i_feel_confident_in_handling_school_challenges.",
    "conflicts_at_home": "home_environment_and_personal_stress_i_experience_conflicts_or_tension_at_home.",
    "current_mode_affects_performance": "learning_modality_and_academic_impact_the_current_mode_of_learning_affects_my_academic_motivation_and_performance.",
    "difficulty_fall_asleep": "sleep_patterns_and_physical_health_i_find_it_difficult_to_fall_asleep_because_of_academic_stress.",
    "emotionally_drained_end_of_day": "emotional_state_and_burnout_indicators_i_feel_emotionally_drained_at_the_end_of_the_school_day.",
    "emotionally_unsupported": "home_environment_and_personal_stress_i_feel_emotionally_unsupported_by_my_family.",
    "family_does_not_understand": "home_environment_and_personal_stress_my_family_does_not_understand_or_acknowledge_my_academic_struggles.",
    "feel_burned_out": "emotional_state_and_burnout_indicators_i_feel_burned_out_even_when_i_try_to_rest.",
    "feel_competent": "motivation_and_personal_accomplishment_i_feel_competent_in_the_subjects_i�m_studying.",
    "feel_disconnected": "social_support_and_isolation_i_feel_disconnected_from_my_academic_community.",
    "feel_helpless": "emotional_state_and_burnout_indicators_i_feel_helpless_when_i_think_about_my_academic_performance.",
    "feel_in_control": "time_management_and_daily_routine_i_feel_in_control_of_how_i_manage_my_time.",
    "feel_isolated": "social_support_and_isolation_i_feel_isolated_or_alone_in_my_academic_journey.",
    "feel_like_giving_up": "emotional_state_and_burnout_indicators_i_sometimes_feel_like_giving_up_on_my_academic_goals.",
    "feel_more_isolated": "learning_modality_and_academic_impact_i_feel_more_isolated_in_online_or_hybrid_learning_environments.",
    "financial_difficulties": "home_environment_and_personal_stress_i_am_currently_experiencing_financial_difficulties_that_affect_my_studies.",
    "finish_before_deadline": "time_management_and_daily_routine_i_usually_finish_tasks_right_before_the_deadline.",
    "frequent_headaches_or_fatigue": "sleep_patterns_and_physical_health_i_experience_frequent_headaches_or_fatigue_during_the_semester.",
    "hard_to_feel_excited": "emotional_state_and_burnout_indicators_i_find_it_hard_to_feel_excited_about_academic_tasks.",
    "hard_to_focus": "emotional_state_and_burnout_indicators_i_find_it_hard_to_feel_excited_about_academic_tasks.",
    "hard_to_maintain_routine": "time_management_and_daily_routine_i_find_it_hard_to_maintain_a_consistent_daily_routine.",
    "hesitate_ask_help": "social_support_and_isolation_i_hesitate_to_ask_for_help_from_peers_or_instructors.",
    "irritated_easily": "emotional_state_and_burnout_indicators_i_get_irritated_or_frustrated_easily_because_of_school_pressure.",
    "learning_setup_stress": "learning_modality_and_academic_impact_my_learning_setup_online_hybrid_or_in_person_contributes_to_my_academic_stress.",
    "long_commutes_stress": "learning_modality_and_academic_impact_long_commutes_to_school_contribute_to_my_stress_and_burnout.",
    "lose_motivation_commute": "learning_modality_and_academic_impact_i_lose_motivation_after_a_long_commute",
    "miss_schoolwork": "home_environment_and_personal_stress_i_sometimes_miss_schoolwork_due_to_family_responsibilities.",
    "multitask_deadlines": "academic_workload_and_study_habits_i_often_multitask_to_meet_academic_deadlines.",
    "noisy_home_environment": "home_environment_and_personal_stress_my_home_environment_is_noisy_or_stressful_making_it_hard_to_study.",
    "not_accomplishing_anything": "motivation_and_personal_accomplishment_i_feel_like_i_am_not_accomplishing_anything_worthwhile_in_school.",
    "personal_life_affects_academics": "home_environment_and_personal_stress_i_feel_that_my_personal_or_home_life_affects_my_academic_performance.",
    "physically_exhausted": "sleep_patterns_and_physical_health_i_often_feel_physically_exhausted_even_after_taking_a_rest.",
    "prefer_current_modality": "learning_modality_and_academic_impact_i_prefer_my_current_learning_modality_over_other_options.",
    "pressure_support_family": "home_environment_and_personal_stress_i_often_feel_pressure_to_support_my_family_financially.",
    "proud_of_achievements": "motivation_and_personal_accomplishment_i_feel_proud_of_my_academic_achievements",
    "question_efforts": "motivation_and_personal_accomplishment_i_often_question_whether_my_efforts_in_school_are_worth_it.",
    "rarely_have_free_time": "academic_workload_and_study_habits_i_rarely_have_free_time_because_of_school_responsibilities.",
    "sacrifice_sleep": "academic_workload_and_study_habits_i_often_sacrifice_sleep_to_finish_schoolwork",
    "sense_of_dread": "emotional_state_and_burnout_indicators_i_feel_a_sense_of_dread_before_starting_school_related_task",
    "skip_meals_due_to_stress": "sleep_patterns_and_physical_health_i_skip_meals_due_to_my_school_workload_or_stress.",
    "sleep_less_than_6_hours": "sleep_patterns_and_physical_health_i_usually_get_less_than_6_hours_of_sleep_on_school_nights.",
    "someone_to_talk": "social_support_and_isolation_i_have_someone_to_talk_to_when_i_feel_burned_out.",
    "struggle_balance_responsibilities": "time_management_and_daily_routine_i_often_struggle_to_balance_school_and_personal_responsibilities.",
    "struggle_organize_tasks": "academic_workload_and_study_habits_i_struggle_to_organize_my_academic_tasks.",
    "study_under_pressure": "academic_workload_and_study_habits_i_usually_study_under_pressure_or_in_last_minute_situations.",
    "supported_by_family": "social_support_and_isolation_i_feel_supported_by_my_family_when_i_feel_stressed_about_school.",
    "traveling_affects_energy": "learning_modality_and_academic_impact_traveling_to_school_affects_my_energy_and_academic_performance._",
    "underperforming_peers": "motivation_and_personal_accomplishment_i_feel_i_am_underperforming_compared_to_my_peers",
    "use_planner": "academic_workload_and_study_habits_i_use_a_planner_or_schedule_to_track_deadlines.",
    "wake_up_tired": "sleep_patterns_and_physical_health_i_often_wake_up_feeling_tired_or_unrefreshed.",
    "waste_time": "time_management_and_daily_routine_i_waste_time_before_starting_my_schoolwork.",
    "workload_heavier_than_peers": "academic_workload_and_study_habits_i_believe_my_workload_is_heavier_than_that_of_my_peers.",
}

# ---------------------------------------------------------
# Utility: Normalize frontend keys
# ---------------------------------------------------------
def normalize_key(key: str) -> str:
    """Convert verbose frontend question text to normalized snake_case keys."""
    key = key.strip().lower()
    key = re.sub(r"[\[\]\(\)\:\?]", "", key)  # remove brackets and punctuation
    key = re.sub(r"[^a-z0-9]+", "_", key)     # replace spaces/symbols with underscores
    key = re.sub(r"_+", "_", key)             # collapse multiple underscores
    return key.strip("_")


# ---------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------
class PredictionPayload(BaseModel):
    """Model for prediction request from frontend."""
    data: Dict[str, Any] = Field(..., description="Survey responses for burnout prediction")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError("data must be a non-empty dictionary")
        return v


# ---------------------------------------------------------
# Required Fields (Based on Your Frontend Survey)
# ---------------------------------------------------------
REQUIRED_FIELDS = [
    # Demographics
    "name",
    "gender",
    "year_level",
    "institution",
    "gwa",
    "how_far_is_your_home_from_school_one_way",
    "learning_modality",

    # Sleep & Physical Health (6 questions)
    "sleep_less_than_6_hours",
    "difficulty_fall_asleep",
    "wake_up_tired",
    "frequent_headaches_or_fatigue",
    "skip_meals_due_to_stress",
    "physically_exhausted",

    # Academic Workload (8 questions)
    "multitask_deadlines",
    "study_under_pressure",
    "academic_workload_unmanageable",
    "rarely_have_free_time",
    "struggle_organize_tasks",
    "use_planner",
    "sacrifice_sleep",
    "workload_heavier_than_peers",

    # Emotional State & Burnout Symptoms (7 questions)
    "emotionally_drained_end_of_day",
    "hard_to_feel_excited",
    "feel_helpless",
    "feel_burned_out",
    "feel_like_giving_up",
    "irritated_easily",
    "sense_of_dread",

    # Motivation & Personal Accomplishment (6 questions)
    "proud_of_achievements",
    "not_accomplishing_anything",
    "confident_handling_challenges",
    "question_efforts",
    "feel_competent",
    "underperforming_peers",

    # Time Management (5 questions)
    "struggle_balance_responsibilities",
    "waste_time",
    "hard_to_maintain_routine",
    "finish_before_deadline",
    "feel_in_control",

    # Social Support (5 questions)
    "supported_by_family",
    "feel_isolated",
    "hesitate_ask_help",
    "someone_to_talk",
    "feel_disconnected",

    # Home Environment (8 questions)
    "financial_difficulties",
    "pressure_support_family",
    "conflicts_at_home",
    "noisy_home_environment",
    "emotionally_unsupported",
    "miss_schoolwork",
    "family_does_not_understand",
    "personal_life_affects_academics",

    # Learning Modality Impact (8 questions)
    "traveling_affects_energy",
    "long_commutes_stress",
    "lose_motivation_commute",
    "hard_to_focus",
    "learning_setup_stress",
    "feel_more_isolated",
    "prefer_current_modality",
    "current_mode_affects_performance",
]


# Optional fields (demographics that might not affect prediction)
OPTIONAL_FIELDS = ["name", "institution"]


def log_frontend_request(raw_data: Dict[str, Any], normalized_data: Dict[str, Any], 
                         mapped_data: Dict[str, Any],
                         endpoint: str, user_context: Optional[Dict[str, Any]] = None):
    """Log frontend requests to Firestore for debugging and analysis."""
    if not db:
        logger.warning("Firestore not available - skipping request logging")
        return None
    
    try:
        log_entry = {
            "endpoint": endpoint,
            "timestamp": datetime.utcnow(),
            "raw_data": raw_data,
            "normalized_data": normalized_data,
            "mapped_data": mapped_data,
            "field_count": len(mapped_data),
            "user_context": user_context or {},
            "status": "received"
        }
        
        doc_ref = db.collection("frontend_requests").add(log_entry)
        request_id = doc_ref[1].id
        
        logger.info(f"📝 Frontend request logged: {request_id}")
        return request_id
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to log frontend request: {e}")
        return None


def validate_and_normalize_payload(raw_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and normalize payload from frontend.
    
    Returns:
        tuple: (normalized_data, mapped_data)
        - normalized_data: Frontend fields with normalized keys
        - mapped_data: Model-ready features with mapped keys
    """
    # Step 1: Normalize keys (frontend format)
    normalized_data = {normalize_key(k): v for k, v in raw_data.items()}
    
    # Step 2: Check for required fields
    missing_fields = []
    for field in REQUIRED_FIELDS:
        if field not in OPTIONAL_FIELDS:
            value = normalized_data.get(field)
            # Check if value is missing or empty
            if value is None or value == "" or value == " ":
                missing_fields.append(field)
    
    if missing_fields:
        logger.warning(f"❌ Missing required fields: {missing_fields}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing required fields",
                "missing_fields": missing_fields,
                "total_missing": len(missing_fields),
                "message": f"Please provide values for all required fields. Missing: {', '.join(missing_fields[:5])}{'...' if len(missing_fields) > 5 else ''}"
            }
        )
    
    # Step 3: Map to model feature names
    mapped_data = {}
    mapping_log = []
    
    for frontend_key, model_key in FIELD_MAPPING.items():
        if frontend_key in normalized_data:
            mapped_data[model_key] = normalized_data[frontend_key]
            mapping_log.append(f"{frontend_key} -> {model_key}")
        else:
            # Try without normalization (in case already normalized)
            if frontend_key in raw_data:
                mapped_data[model_key] = raw_data[frontend_key]
                mapping_log.append(f"{frontend_key} -> {model_key} (direct)")
    
    # Log mapping details
    logger.info(f"📊 Field Mapping Summary:")
    logger.info(f"   Frontend fields: {len(normalized_data)}")
    logger.info(f"   Mapped features: {len(mapped_data)}")
    logger.info(f"   Required fields present: {len([f for f in REQUIRED_FIELDS if f in normalized_data])}")
    
    # Log first few mappings for verification
    logger.debug(f"   Sample mappings:")
    for log_entry in mapping_log[:5]:
        logger.debug(f"      {log_entry}")
    
    return normalized_data, mapped_data


# ---------------------------------------------------------
# /predict - Main prediction endpoint
# ---------------------------------------------------------
@router.post("/", 
             summary="Predict burnout risk from survey responses",
             status_code=status.HTTP_200_OK,
             response_description="Burnout prediction with detailed analysis")
async def predict(payload: PredictionPayload):
    """
    Predict burnout level from survey responses.
    
    This endpoint:
    1. Validates all required survey fields
    2. Normalizes input data (frontend format)
    3. Maps to model feature names
    4. Runs prediction using active ML model
    5. Returns detailed analysis with recommendations
    6. Logs all data to Firestore
    
    Returns:
    - burnout_level: Low/Moderate/High
    - confidence: Prediction confidence score
    - risk_score: Quantitative risk assessment (0-100)
    - explanation: Feature analysis and contributing factors
    - recommendations: Personalized intervention suggestions
    """
    try:
        logger.info("=" * 80)
        logger.info("NEW PREDICTION REQUEST")
        logger.info("=" * 80)
        
        raw_data = payload.data
        user_context = {
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "endpoint": "/predict",
            "request_timestamp": datetime.utcnow().isoformat()
        }
        
        normalized_data, mapped_data = validate_and_normalize_payload(raw_data)
        
        logger.info(f"✅ Validation passed")
        logger.info(f"   Frontend fields: {len(normalized_data)}")
        logger.info(f"   Mapped features: {len(mapped_data)}")
        
        # Log sample of mapped data
        logger.info(f"📝 Sample Mapped Data (first 10 features):")
        sample_features = list(mapped_data.items())
        for feature, value in sample_features:
            logger.info(f"   {feature}: {value}")
        
        # Log request to Firestore
        request_id = log_frontend_request(
            raw_data=raw_data,
            normalized_data=normalized_data,
            mapped_data=mapped_data,
            endpoint="/predict",
            user_context=user_context
        )
        
        # Run prediction with mapped data
        logger.info("🚀 Running prediction with mapped features...")
        prediction_result = run_prediction(
            payload=mapped_data,  # Use mapped data instead of normalized
            log_to_firestore=True,
            user_context=user_context
        )
        
        if not prediction_result.get("success", False):
            logger.error(f"❌ Prediction failed: {prediction_result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Prediction failed",
                    "message": prediction_result.get("message", "An error occurred during prediction"),
                    "details": prediction_result.get("error")
                }
            )
        
        logger.info(f"✅ Prediction successful: {prediction_result.get('burnout_level')} "
                   f"(confidence: {prediction_result.get('probability', 0)*100:.1f}%)")
        
        # Build response
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "request_id": request_id,
            "prediction": prediction_result,
            "metadata": {
                "fields_received": len(raw_data),
                "fields_normalized": len(normalized_data),
                "features_mapped": len(mapped_data),
                "model_version": prediction_result.get("model_info", {}).get("version"),
                "processing_time": prediction_result.get("prediction_metadata", {}).get("processing_time_seconds")
            }
        }
        
        logger.info("=" * 80)
        logger.info("✅ PREDICTION COMPLETE")
        logger.info("=" * 80)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Unexpected error in prediction endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred during prediction",
                "details": str(e)
            }
        )


# ---------------------------------------------------------
# /health - Service health check
# ---------------------------------------------------------
@router.get("/health",
            summary="Check prediction service health",
            status_code=status.HTTP_200_OK)
async def health_check():
    """
    Comprehensive health check for prediction service.
    
    Checks:
    - Firestore connectivity
    - Active model availability
    - Model loading capability
    - System readiness
    """
    try:
        health_status = check_model_health()
        
        # Check Firestore
        firestore_ok = False
        if db is not None:
            try:
                # Try a simple query
                _ = list(db.collection("models").limit(1).stream())
                firestore_ok = True
            except Exception as e:
                logger.warning(f"Firestore health check failed: {e}")
        
        overall_status = "healthy" if (
            health_status.get("status") == "healthy" and firestore_ok
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "components": {
                "model": health_status.get("status"),
                "firestore": "connected" if firestore_ok else "disconnected",
                "prediction_service": "operational" if health_status.get("status") == "healthy" else "degraded"
            },
            "details": health_status,
            "message": health_status.get("message", "Service status unknown")
        }
        
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "message": f"Health check error: {str(e)}"
        }


# ---------------------------------------------------------
# /debug - Debug endpoint for development
# ---------------------------------------------------------
@router.post("/debug",
             summary="Debug prediction pipeline (development only)",
             status_code=status.HTTP_200_OK,
             include_in_schema=False)
async def debug_prediction(payload: PredictionPayload):
    """
    Debug endpoint to inspect data transformation pipeline.
    
    ⚠️ FOR DEVELOPMENT USE ONLY - DO NOT USE IN PRODUCTION
    
    Returns detailed information about:
    - Raw input data
    - Normalized data
    - Mapped features
    - Field validation
    - Data types
    """
    try:
        raw_data = payload.data
        normalized_data = {normalize_key(k): v for k, v in raw_data.items()}
        
        # Apply mapping
        mapped_data = {}
        for frontend_key, model_key in FIELD_MAPPING.items():
            if frontend_key in normalized_data:
                mapped_data[model_key] = normalized_data[frontend_key]
        
        # Analyze fields
        field_analysis = {}
        for key, value in normalized_data.items():
            mapped_key = FIELD_MAPPING.get(key, "NOT_MAPPED")
            field_analysis[key] = {
                "value": value,
                "type": type(value).__name__,
                "is_required": key in REQUIRED_FIELDS,
                "is_empty": value in (None, "", " "),
                "mapped_to": mapped_key,
                "length": len(str(value)) if value is not None else 0
            }
        
        # Check which required fields are present
        present_required = [f for f in REQUIRED_FIELDS if f in normalized_data]
        missing_required = [f for f in REQUIRED_FIELDS if f not in normalized_data and f not in OPTIONAL_FIELDS]
        
        return {
            "debug_info": {
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "raw_field_count": len(raw_data),
                "normalized_field_count": len(normalized_data),
                "mapped_field_count": len(mapped_data),
                "required_fields_total": len(REQUIRED_FIELDS),
                "required_fields_present": len(present_required),
                "required_fields_missing": len(missing_required),
            },
            "field_analysis": field_analysis,
            "mapped_features_sample": dict(list(mapped_data.items())[:10]),
            "required_fields_status": {
                "present": present_required[:10],
                "missing": missing_required[:10],
                "total_present": len(present_required),
                "total_missing": len(missing_required)
            },
            "mapping_verification": {
                "sample_mappings": [
                    {
                        "frontend": k,
                        "normalized": normalize_key(k),
                        "model_feature": FIELD_MAPPING.get(normalize_key(k), "NOT_MAPPED")
                    }
                    for k in list(raw_data.keys())[:5]
                ]
            },
            "warnings": [
                "⚠️ This is a debug endpoint",
                "⚠️ Do not use in production",
                "⚠️ Contains sensitive data inspection"
            ]
        }
        
    except Exception as e:
        logger.exception("Debug endpoint error")
        return {
            "error": str(e),
            "traceback": str(e)
        }


# ---------------------------------------------------------
# /fields - Return required field list
# ---------------------------------------------------------
@router.get("/fields",
            summary="Get list of required survey fields",
            status_code=status.HTTP_200_OK)
async def get_required_fields():
    """
    Return the list of required survey fields for frontend validation.
    
    Use this endpoint to:
    - Validate frontend forms
    - Ensure complete data submission
    - Match backend expectations
    - Understand field mapping
    """
    return {
        "required_fields": REQUIRED_FIELDS,
        "optional_fields": OPTIONAL_FIELDS,
        "field_mapping": FIELD_MAPPING,
        "total_required": len([f for f in REQUIRED_FIELDS if f not in OPTIONAL_FIELDS]),
        "field_categories": {
            "demographics": 7,
            "sleep_health": 6,
            "academic_workload": 8,
            "emotional_state": 7,
            "motivation": 6,
            "time_management": 5,
            "social_support": 5,
            "home_environment": 8,
            "learning_modality": 8
        },
        "total_questions": len(REQUIRED_FIELDS),
        "note": "Frontend fields will be normalized and mapped to model feature names automatically"
    }