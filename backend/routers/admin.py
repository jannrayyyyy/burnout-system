from fastapi import APIRouter, HTTPException
from backend.services.data_service import export_responses_csv, fetch_all_responses
from backend.services.model_service import retrain_from_dataframe, MODEL_PATH
import pandas as pd
from datetime import datetime


ADMIN_SECRET = os.getenv("ADMIN_SECRET", None)

router = APIRouter()

@router.post("/admin/retrain")
def retrain_model():
    rows = fetch_all_responses()
    if not rows:
        raise HTTPException(status_code=400, detail="No data available to retrain")

    df = pd.DataFrame([r["data"] | {"prediction": r.get("prediction")} for r in rows if "data" in r])
    count = retrain_from_dataframe(df)

    return {"message": "Model retrained", "records_used": count, "retrained_at": datetime.utcnow().isoformat()}

@router.get("/admin/export")
def export_csv():
    resp = export_responses_csv()
    if not resp:
        raise HTTPException(status_code=400, detail="No data found")
    return resp

@router.get("/admin/info")
def model_info():
    return {
        "last_trained": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
        "records_in_db": len(fetch_all_responses())
    }
