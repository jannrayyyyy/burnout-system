# backend/routers/admin.py
from fastapi import APIRouter, HTTPException, Query, Header
from backend.services.data_service import export_responses_csv, fetch_all_responses
from backend.services.model_service import retrain_from_dataframe, MODEL_PATH, list_versions_from_files, rollback_to_version
from backend.services.firebase_service import db
import pandas as pd
from datetime import datetime
import os

router = APIRouter()

ADMIN_SECRET = os.getenv("ADMIN_SECRET", None)

def _require_admin(x_admin_secret: str = Header(None)):
    if ADMIN_SECRET and x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

@router.post("/admin/retrain")
def retrain_model(description: str = Query("Retrained via admin panel", description="Description of changes"), x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    rows = fetch_all_responses()
    if not rows:
        raise HTTPException(status_code=400, detail="No data available to retrain")

    df = pd.DataFrame([r["data"] | {"prediction": r.get("prediction")} for r in rows if "data" in r])
    version, count = retrain_from_dataframe(df, description=description)

    return {
        "message": "Model retrained",
        "version": version,
        "records_used": count,
        "retrained_at": datetime.utcnow().isoformat(),
        "description": description,
    }

@router.get("/admin/export")
def export_csv(x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    resp = export_responses_csv()
    if not resp:
        raise HTTPException(status_code=400, detail="No data found")
    return resp

@router.get("/admin/info")
def model_info():
    last_trained = None
    if MODEL_PATH.exists():
        last_trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat()
    return {
        "last_trained": last_trained,
        "records_in_db": len(fetch_all_responses()),
    }

@router.get("/admin/models")
def list_models():
    # Prefer listing metadata from Firestore if available
    models_meta = []
    try:
        docs = db.collection("models").order_by("created_at", direction= "DESCENDING").stream()
        for doc in docs:
            d = doc.to_dict()
            d["id"] = doc.id
            models_meta.append(d)
    except Exception:
        # fallback to local files
        files = list_versions_from_files()
        for f in files:
            models_meta.append({
                "version": f.get("version"),
                "file": f.get("file"),
                "created_at": datetime.fromtimestamp(f.get("modified")).isoformat()
            })
    return {"count": len(models_meta), "items": models_meta}

@router.post("/admin/rollback/{version}")
def rollback(version: int, x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    try:
        v = rollback_to_version(int(version))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": "Rolled back", "version": v, "rolled_back_at": datetime.utcnow().isoformat()}


@router.get("/admin/current")
def current_model():
    info = get_current_model_info()
    if not info:
        raise HTTPException(status_code=404, detail="No active model found")
    return info

@router.get("/admin/data/{version}")
def model_data(version: int, x_admin_secret: str = Header(None)):
    _require_admin(x_admin_secret)
    """
    Fetch survey data (training data) used for a given version.
    This will just fetch all responses until now — since retrain uses all responses.
    In the future, you can store exact dataset snapshot per version.
    """
    rows = fetch_all_responses()
    if not rows:
        raise HTTPException(status_code=404, detail="No data available")

    df = pd.DataFrame([r["data"] | {"prediction": r.get("prediction")} for r in rows if "data" in r])
    # only return a preview
    preview = df.head(50).to_dict(orient="records")
    return JSONResponse({"version": version, "preview_count": len(preview), "preview": preview})