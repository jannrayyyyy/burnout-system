from backend.services.firebase_service import db
import pandas as pd

def fetch_all_responses():
    docs = db.collection("survey_responses").stream()
    return [doc.to_dict() for doc in docs]

def export_responses_csv():
    rows = fetch_all_responses()
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df.to_dict(orient="records")
