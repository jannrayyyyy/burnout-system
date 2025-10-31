# backend/routers/admin.py
from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import os
import logging
from google.cloud.firestore import DocumentSnapshot
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

from backend.services.data_service import export_responses_csv, fetch_all_responses
from backend.services.training_service import train_from_csv  # <-- new training logic here
from backend.services.firebase_service import db

from backend.services.data_service import export_responses_csv, fetch_all_responses
from backend.services.model_service import (
    retrain_from_dataframe,
    MODEL_PATH,
    list_versions_from_files,
    rollback_to_version,
    add_survey_without_prediction,
    get_current_model_info,
    hard_reset,
)

from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

# ===============================
# 🔐 Admin Auth Helper
# ===============================
ADMIN_SECRET = os.getenv("ADMIN_SECRET", None)

def _require_admin(x_admin_secret: str = Header(None)):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


# ===============================
# 🧠 Train Endpoint (one-shot)
# ===============================
@router.post("/admin/train")
def train_model(
    description: str = Query("Initial train via admin panel", description="Description for model metadata"),
    x_admin_secret: str = Header(None),
):
    _require_admin(x_admin_secret)
    try:
        result = train_from_csv(description=description)
        return {
            "message": "Model trained successfully",
            "version": result["version"],
            "records_used": result["records_used"],
            "accuracy": result["accuracy"],
            "passed": result["passed"],
            "trained_at": datetime.utcnow().isoformat(),
            "description": description,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Unexpected error in /admin/train")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===============================
# 🧠 Retrain Endpoint
# ===============================
@router.post("/admin/retrain")
def retrain_model(
    description: str = Query("Retrained via admin panel", description="Description of changes"),
    x_admin_secret: str = Header(None),
):
    _require_admin(x_admin_secret)

    # Fetch all available responses (both trained + untrained)
    rows = fetch_all_responses()
    if not rows:
        raise HTTPException(status_code=400, detail="No data available to retrain")

    # Normalize all rows into a consistent structure
    data_list = []
    for r in rows:
        if not isinstance(r, dict):
            continue

        # Handle both "data" subfield or top-level data
        if "data" in r and isinstance(r["data"], dict):
            entry = r["data"].copy()
        else:
            entry = {k: v for k, v in r.items() if k != "id"}

        # Attach prediction if available
        if "prediction" in r:
            entry["prediction"] = r.get("prediction")

        data_list.append(entry)

    if not data_list:
        raise HTTPException(status_code=400, detail="No valid data entries found for retraining")

    # Convert to DataFrame safely
    try:
        df = pd.DataFrame(data_list)
        if df.empty:
            raise ValueError("Empty DataFrame after parsing rows")
    except Exception as e:
        logger.exception("Failed to build DataFrame for retraining")
        raise HTTPException(status_code=500, detail=f"Data parsing failed: {e}")

    # Attempt retrain
    try:
        version, count = retrain_from_dataframe(df, description=description)
    except Exception as e:
        logger.exception("Failed to retrain model")
        raise HTTPException(status_code=500, detail=str(e))

    # Return success response
    return {
        "message": "Model retrained successfully",
        "version": version,
        "records_used": count,
        "retrained_at": datetime.utcnow().isoformat(),
        "description": description,
    }


# ===============================
# Export Data
# ===============================
@router.get("/admin/export")
def export_csv(upload_to_storage: bool = Query(False), x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    resp = export_responses_csv(upload_to_storage=upload_to_storage)
    if not resp:
        raise HTTPException(status_code=400, detail="No data found")
    # If file_response is returned
    if "file_response" in resp:
        return resp["file_response"]
    else:
        return resp


# ===============================
# Model Info
# ===============================
@router.get("/admin/info")
def model_info(x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    last_trained = None
    if MODEL_PATH.exists():
        last_trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat()
    return {
        "last_trained": last_trained,
        "records_in_db": len(fetch_all_responses()),
    }


# ===============================
# 📦 List Models
# ===============================

@router.get("/admin/models")
def list_models(x_admin_secret: str = Header(None)):
    """
    List all trained models from Firestore with full JSON safety.
    Handles NaN/Inf floats, Firestore Timestamps, and nested structures in sample_preview.
    Falls back to local model files if Firestore query fails.
    """
    _require_admin(x_admin_secret)
    models_meta = []

    import math

    def _safe_value(v):
        """Recursively convert any Firestore / Python value into JSON-safe form."""
        # Basic types
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(v, datetime):
            return v.isoformat()
        if hasattr(v, "isoformat"):  # Firestore Timestamp
            return v.isoformat()
        if isinstance(v, (list, tuple)):
            return [_safe_value(i) for i in v]
        if isinstance(v, dict):
            return {k: _safe_value(i) for k, i in v.items()}
        return v

    try:
        docs = db.collection("models").order_by("created_at", direction="DESCENDING").stream()

        for doc in docs:
            d = doc.to_dict()
            d["id"] = doc.id

            # Ensure all required fields exist (fill missing with defaults)
            d.setdefault("created_at", datetime.utcnow().isoformat())
            d.setdefault("description", "")
            d.setdefault("file", "")
            d.setdefault("records_used", 0)
            d.setdefault("sample_preview", [])
            d.setdefault("source_collection", "")
            d.setdefault("training_type", "unsupervised")
            d.setdefault("version", 0)

            # Make all values JSON-safe
            safe_doc = {k: _safe_value(v) for k, v in d.items()}
            models_meta.append(safe_doc)

    except Exception as e:
        logger.exception(f"Error fetching Firestore models: {e}")
        # fallback to local files
        from pathlib import Path
        files = list_versions_from_files()
        for f in files:
            models_meta.append({
                "version": int(f.get("version")),
                "file": str(f.get("file")),
                "created_at": datetime.fromtimestamp(f.get("modified")).isoformat(),
                "description": "Local file model",
                "records_used": 0,
                "sample_preview": [],
                "source_collection": "local",
                "training_type": "unknown"
            })

    # Double-sanitize and validate JSON
    try:
        import json
        json.dumps(models_meta, allow_nan=False)
    except ValueError:
        # Replace any lingering invalid floats
        for m in models_meta:
            for k, v in m.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    m[k] = None

    return {"count": len(models_meta), "items": models_meta}


# ===============================
# 🔄 Rollback
# ===============================
@router.post("/admin/rollback/{version}")
def rollback(version: int, x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    try:
        v = rollback_to_version(int(version))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": "Rolled back", "version": v, "rolled_back_at": datetime.utcnow().isoformat()}


# ===============================
# 🔍 Current Model
# ===============================
@router.get("/admin/current")
def current_model(x_admin_secret: str = Header(None)):
    """
    Return current active model metadata, safely JSON-compliant.
    If no model is found, returns an empty payload instead of error.
    """
    _require_admin(x_admin_secret)
    info = get_current_model_info()

    import math
    from datetime import datetime

    def _safe_value(v):
        """Convert nested model info to JSON-safe primitives."""
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(v, datetime):
            return v.isoformat()
        if hasattr(v, "isoformat"):  # Firestore Timestamp
            return v.isoformat()
        if isinstance(v, (list, tuple)):
            return [_safe_value(i) for i in v]
        if isinstance(v, dict):
            return {k: _safe_value(i) for k, i in v.items()}
        return v

    if not info:
        return {"message": "No active model found"}

    try:
        safe_info = {k: _safe_value(v) for k, v in info.items()}
    except Exception:
        safe_info = {"message": "Invalid model metadata"}

    # Validate JSON compliance
    import json
    try:
        json.dumps(safe_info, allow_nan=False)
    except ValueError:
        for k, v in list(safe_info.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                safe_info[k] = None

    return safe_info

# ===============================
# 📂 Model Data Preview
# ===============================
@router.get("/admin/data/{version}")
def model_data(version: int, x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    rows = fetch_all_responses()
    if not rows:
        raise HTTPException(status_code=404, detail="No data available")

    # Build DataFrame safely: each row may have 'data' nested dict; include prediction if present
    records = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if "data" in r and isinstance(r["data"], dict):
            rec = r["data"].copy()
        else:
            rec = {k: v for k, v in r.items() if k != "id"}
        if "prediction" in r:
            rec["prediction"] = r.get("prediction")
        records.append(rec)

    if not records:
        return JSONResponse({"version": version, "preview_count": 0, "preview": []})

    df = pd.DataFrame(records)
    preview = df.head(50).to_dict(orient="records")
    return JSONResponse({"version": version, "preview_count": len(preview), "preview": preview})
# ===============================
# 📝 Add Survey (without prediction)
# ===============================
class InputPayload(BaseModel):
    data: dict

@router.post("/admin/submit")
def submit(payload: InputPayload, x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    result = add_survey_without_prediction(payload.data)
    return {"message": "Survey added (no prediction)", **result}


# ===============================
# 💣 Hard Reset Endpoint
# ===============================
@router.post("/admin/reset")
def reset_all(x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    result = hard_reset()
    return result
