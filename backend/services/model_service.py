# backend/services/model_service.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from backend.services.firebase_service import db

# paths
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "burnout_pipeline.pkl"

# ensure MODELS_DIR exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# load pipeline (latest)
clf_pipeline = None
if MODEL_PATH.exists():
    clf_pipeline = joblib.load(MODEL_PATH)
else:
    # do not raise here; other code can create model via retrain
    clf_pipeline = None

def run_prediction(data: dict):
    if clf_pipeline is None:
        raise RuntimeError("No model loaded. Train or deploy a model first.")
    df = pd.DataFrame([data])
    proba = clf_pipeline.predict_proba(df)[0]
    classes = clf_pipeline.classes_
    best_idx = int(proba.argmax())
    best_class = str(classes[best_idx])
    best_prob = float(proba[best_idx])
    return {
        "burnout_level": best_class,
        "probability": best_prob,
        "all_probabilities": dict(zip(map(str, classes.tolist()), proba.tolist()))
    }

def _build_pipeline():
    # This must match your original training pipeline preprocessing
    categorical = ["gender"]
    # numeric will be autodetected when fitting
    def make_pipeline():
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", "passthrough", [col for col in []])  # placeholder; sklearn handles passthrough shape during fit
            ],
            remainder="passthrough"  # allow numeric columns to pass through
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        return pipeline
    return make_pipeline()

def retrain_from_dataframe(df: pd.DataFrame, description: str = "Retrained model"):
    """
    Expects df to contain all features (same columns used in training) and a 'prediction' column
    """
    if "prediction" not in df.columns:
        raise ValueError("Missing 'prediction' column for retraining")

    X = df.drop(columns=["prediction"]).copy()
    y = df["prediction"].copy()

    # Ensure columns order and types are consistent; OneHotEncoder handles categories
    # Build and fit pipeline
    pipeline = _build_pipeline()
    # sklearn will accept DataFrame with object/string columns (OneHotEncoder will handle categorical)
    pipeline.fit(X, y)

    # versioning
    existing = sorted(MODELS_DIR.glob("burnout_v*.pkl"))
    version = 1 if not existing else (len(existing) + 1)
    version_file = MODELS_DIR / f"burnout_v{version}.pkl"

    # save versioned and latest
    joblib.dump(pipeline, version_file)
    joblib.dump(pipeline, MODEL_PATH)

    # update global
    global clf_pipeline
    clf_pipeline = pipeline

    # save metadata in firebase (collection: models)
    meta = {
        "version": version,
        "file": version_file.name,
        "created_at": datetime.utcnow().isoformat(),
        "records_used": len(df),
        "description": description,
    }
    db.collection("models").add(meta)

    return version, len(df)

def list_versions_from_files():
    """
    Returns list of versions found in models/ folder (local files) sorted desc
    """
    items = []
    for p in sorted(MODELS_DIR.glob("burnout_v*.pkl"), reverse=True):
        # derive version number
        name = p.name
        try:
            v = int(name.replace("burnout_v", "").replace(".pkl", ""))
        except Exception:
            v = None
        items.append({"file": name, "version": v, "path": str(p), "modified": p.stat().st_mtime})
    return items

def rollback_to_version(version: int):
    """
    Load a versioned model file (burnout_v{version}.pkl) and make it the latest.
    """
    version_file = MODELS_DIR / f"burnout_v{version}.pkl"
    if not version_file.exists():
        raise FileNotFoundError(f"Version file not found: {version_file}")

    pipeline = joblib.load(version_file)
    # overwrite latest
    joblib.dump(pipeline, MODEL_PATH)

    # update global
    global clf_pipeline
    clf_pipeline = pipeline

    # log rollback in firebase
    db.collection("models").add({
        "action": "rollback",
        "version": version,
        "file": version_file.name,
        "created_at": datetime.utcnow().isoformat(),
        "note": f"Rolled back to version {version}"
    })

    return version

def get_current_model_info():
    """
    Returns info about the current active model (burnout_pipeline.pkl).
    """
    if not MODEL_PATH.exists():
        return None

    # Try to match with Firestore metadata
    docs = db.collection("models").order_by("created_at", direction="DESCENDING").stream()
    for doc in docs:
        d = doc.to_dict()
        # if this model is latest and has file == burnout_v{n}.pkl
        if "file" in d:
            return {
                "version": d.get("version"),
                "file": d.get("file"),
                "created_at": d.get("created_at"),
                "records_used": d.get("records_used"),
                "description": d.get("description", "-"),
                "note": d.get("note", None)
            }

    # fallback: use only file info
    return {
        "version": None,
        "file": MODEL_PATH.name,
        "created_at": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
        "records_used": None,
        "description": "Unknown (no metadata)"
    }
