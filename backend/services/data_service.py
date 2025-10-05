# backend/services/data_service.py
from backend.services.firebase_service import db, bucket
import pandas as pd
import tempfile
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

def fetch_all_responses():
    """
    Returns all documents in `untrained_surveys` as a list of dicts.
    """
    docs = db.collection("untrained_surveys").stream()
    return [doc.to_dict() for doc in docs]

def fetch_untrained_documents():
    """
    Returns list of (doc_id, data) for untrained_surveys.
    """
    docs = db.collection("untrained_surveys").stream()
    return [(doc.id, doc.to_dict()) for doc in docs]

def export_responses_csv(upload_to_storage: bool = False) -> Optional[dict]:
    """
    Exports survey_responses collection to CSV. If upload_to_storage True and a Firebase
    bucket is configured, uploads file and returns url.
    Returns:
      - None if no rows
      - {"file_response": FileResponse} OR {"storage_url": "https://..."} on success
    """
    try:
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

        # If upload to storage requested and configured, upload and return public URL
        if upload_to_storage and bucket:
            filename = Path(tmp.name).name
            blob = bucket.blob(f"exports/{filename}")
            blob.upload_from_filename(tmp.name)
            try:
                # Make public (note: may not be ideal for security; change to signed URL if needed)
                blob.make_public()
                public_url = blob.public_url
            except Exception:
                public_url = None
            # Clean temp file
            try:
                os.remove(tmp.name)
            except Exception:
                logger.exception("Failed to remove temp CSV file")

            return {"storage_url": public_url, "filename": filename}

        # Otherwise return a FileResponse (useful for direct download via API)
        return {"file_response": FileResponse(tmp.name, filename="survey_responses.csv", media_type="text/csv")}
    except Exception as e:
        logger.exception("Failed to export responses to CSV")
        raise

