# backend/services/firebase_service.py
import os
import base64
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore

# If Render supplies a base64-encoded JSON key in FIREBASE_KEY_BASE64, write it to backend/firebase_key.json
KEY_PATH = Path(__file__).resolve().parent.parent / "firebase_key.json"
key_b64 = os.getenv("FIREBASE_KEY_BASE64")
if key_b64 and not KEY_PATH.exists():
    decoded = base64.b64decode(key_b64.encode("utf-8"))
    with open(KEY_PATH, "wb") as f:
        f.write(decoded)

if not KEY_PATH.exists():
    raise RuntimeError("Missing firebase_key.json and FIREBASE_KEY_BASE64 is not set.")

cred = credentials.Certificate(str(KEY_PATH))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()
