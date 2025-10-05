# backend/services/firebase_service.py
import os
import base64
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore, storage

# write FIREBASE_KEY_BASE64 env to backend/firebase_key.json if provided (useful for Render)
KEY_PATH = Path(__file__).resolve().parent.parent / "firebase_key.json"
key_b64 = os.getenv("FIREBASE_KEY_BASE64")
if key_b64 and not KEY_PATH.exists():
    decoded = base64.b64decode(key_b64.encode("utf-8"))
    KEY_PATH.write_bytes(decoded)

if not KEY_PATH.exists():
    raise RuntimeError("Missing firebase_key.json. Place it in backend/ or set FIREBASE_KEY_BASE64 env.")

cred = credentials.Certificate(str(KEY_PATH))
# Initialize app only once
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Storage (optional)
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")  # e.g. "your-bucket.appspot.com"
bucket = None
if FIREBASE_STORAGE_BUCKET:
    try:
        bucket = storage.bucket(FIREBASE_STORAGE_BUCKET)
    except Exception as e:
        # If bucket initialization fails, keep bucket as None but log
        # (Avoid failing startup if user doesn't need storage)
        print(f"⚠️ Could not initialize Firebase Storage bucket: {e}")
        bucket = None
