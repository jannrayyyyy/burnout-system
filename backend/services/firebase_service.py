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

# Allow overriding key path via env
env_key_path = os.getenv("FIREBASE_KEY_PATH")
if env_key_path:
    env_p = Path(env_key_path)
    if env_p.exists():
        KEY_PATH = env_p

# If key is missing, don't crash app at import time. Downstream code should handle db==None.
if KEY_PATH.exists():
    cred = credentials.Certificate(str(KEY_PATH))
    # Initialize app only once
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    print("⚠️ firebase_key.json not found and FIREBASE_KEY_BASE64 not set. Firebase disabled.")
    db = None

# Storage (optional)
FIREBASE_STORAGE_BUCKET = "burnout-system.firebasestorage.app"  # e.g. "your-bucket.appspot.com"
bucket = None
if FIREBASE_STORAGE_BUCKET and db is not None:
    try:
        bucket = storage.bucket(FIREBASE_STORAGE_BUCKET)
    except Exception as e:
        # If bucket initialization fails, keep bucket as None but log
        # (Avoid failing startup if user doesn't need storage)
        print(f"⚠️ Could not initialize Firebase Storage bucket: {e}")
        bucket = None
