# backend/services/data_service.py
from backend.services.firebase_service import db
import pandas as pd
import tempfile
from fastapi.responses import FileResponse

def fetch_all_responses():
    docs = db.collection("survey_responses").stream()
    return [doc.to_dict() for doc in docs]

def export_responses_csv():
    rows = []
    docs = db.collection("survey_responses").stream()
    for doc in docs:
        d = doc.to_dict()
        # flatten: include 'prediction' at top-level if present
        if "data" in d:
            r = d["data"].copy()
            r["prediction"] = d.get("prediction")
            r["created_at"] = d.get("created_at")
            rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return FileResponse(tmp.name, filename="survey_responses.csv", media_type="text/csv")
