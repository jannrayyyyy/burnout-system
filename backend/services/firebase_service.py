import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path

FIREBASE_KEY_PATH = Path(__file__).resolve().parent.parent / "firebase_key.json"

if not FIREBASE_KEY_PATH.exists():
    raise RuntimeError("Missing firebase_key.json. Place it in backend/")

cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
firebase_admin.initialize_app(cred)
db = firestore.client()
