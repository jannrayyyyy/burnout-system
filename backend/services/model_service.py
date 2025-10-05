# backend/services/model_service.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from backend.services.firebase_service import db
from backend.services.data_service import fetch_untrained_documents
import logging
from typing import Tuple, Optional, List

from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# paths
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "burnout_pipeline.pkl"

# ensure MODELS_DIR exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# load pipeline (latest)
clf_pipeline: Optional[Pipeline] = None
if MODEL_PATH.exists():
    try:
        clf_pipeline = joblib.load(MODEL_PATH)
    except Exception:
        logger.exception("Failed to load existing model file.")
        clf_pipeline = None
else:
    clf_pipeline = None


# ---------------------
# Helpers: pipeline
# ---------------------
def _build_pipeline() -> Pipeline:
    """
    Build the ML pipeline. Compatible with both old and new sklearn versions.
    """
    categorical = ["gender"]

    # handle sklearn>=1.2 (sparse_output) and older (sparse)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # fallback for older sklearn versions
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[("cat", encoder, categorical)],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    return pipeline


# ---------------------
# 1) Prediction
# ---------------------
def run_prediction(data: dict):
    global clf_pipeline
    if clf_pipeline is None:
        raise RuntimeError("No model loaded. Train or deploy a model first.")
    df = pd.DataFrame([data])
    proba = clf_pipeline.predict_proba(df)[0]
    classes = clf_pipeline.classes_
    best_idx = int(proba.argmax())
    best_class = str(classes[best_idx])
    best_prob = float(proba[best_idx])

    # log survey into untrained_surveys for later training review
    db.collection("untrained_surveys").add({
        **data,
        "burnout_level": best_class,
        "probability": best_prob,
        "created_at": datetime.utcnow().isoformat(),
        "status": "predicted"
    })

    return {
        "burnout_level": best_class,
        "probability": best_prob,
        "all_probabilities": dict(zip(map(str, classes.tolist()), proba.tolist()))
    }


# ---------------------
# 2) Train from untrained_surveys (one-shot)
# ---------------------
def train_from_untrained(description: str = "Initial trained model") -> Tuple[int, int]:
    """
    Train a model using data from `untrained_surveys`.
    - If no model exists, bootstraps an initial unsupervised model using KMeans clustering.
    - If labels (prediction/burnout_level) exist, performs supervised training.
    - If a model already exists, refuses and instructs to use retrain.
    Returns (version, records_used).
    """
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import KMeans

    global clf_pipeline

    # --- prevent retraining over existing model
    if MODEL_PATH.exists():
        raise RuntimeError("A model already exists. Use retrain endpoint instead of train.")

    # --- fetch untrained data
    docs = fetch_untrained_documents()  # (doc_id, data)
    if not docs:
        raise ValueError("No untrained surveys available to train a model.")

    records = []
    for _id, data in docs:
        if "data" in data and isinstance(data["data"], dict):
            r = data["data"].copy()
            if "prediction" in data:
                r["prediction"] = data.get("prediction")
            elif "burnout_level" in data:
                r["burnout_level"] = data.get("burnout_level")
        else:
            r = data.copy()
        records.append(r)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Untrained data is empty or invalid format.")

    # ---------------------
    # CASE 1: SUPERVISED
    # ---------------------
    label_cols = [c for c in ["prediction", "label", "burnout_level"] if c in df.columns]
    if label_cols:
        label_col = label_cols[0]
        X = df.drop(columns=[label_col]).copy()
        y = df[label_col].astype(str).copy()
        logger.info(f"Training supervised model using label column: {label_col}")

    else:
        # ---------------------
        # CASE 2: UNSUPERVISED (KMeans Clustering)
        # ---------------------
        logger.warning("No label column found — bootstrapping model using KMeans clustering.")

        # Clean and filter data
        df = df.copy()

        # Drop obviously non-feature fields
        drop_cols = [
            "created_at", "status", "timestamp", "trained_at", "model_version",
            "id", "burnout_level", "prediction", "label"
        ]
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Convert all possible numeric columns safely
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        # Drop all-NaN columns
        numeric_df = numeric_df.dropna(axis=1, how="all")

        if numeric_df.empty:
            raise ValueError(
                "Cannot perform unsupervised training: no numeric fields found. "
                "Ensure surveys include numeric values like 'age', 'hours_online', 'gwa', etc."
            )

        # Fill NaNs with median
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        numeric_df = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

        # Drop constant columns
        numeric_df = numeric_df.loc[:, numeric_df.apply(pd.Series.nunique) > 1]
        if numeric_df.empty:
            raise ValueError("Cannot perform unsupervised training: all numeric fields have constant values.")

        # Determine cluster count
        import numpy as np
        from sklearn.cluster import KMeans
        unique_rows = np.unique(numeric_df, axis=0)
        n_clusters = min(3, len(unique_rows))
        if n_clusters < 2:
            raise ValueError("Not enough unique numeric data to form clusters.")

        # Run KMeans
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            clusters = kmeans.fit_predict(numeric_df)
        except Exception as e:
            raise RuntimeError(f"KMeans clustering failed: {str(e)}")

        # Map clusters → burnout levels
        cluster_map = {0: "low", 1: "medium", 2: "high"}
        try:
            means = numeric_df.groupby(clusters).mean()
            ref_col = next((c for c in ["perceived_stress", "procrastination", "hours_online"] if c in means.columns), None)
            if ref_col:
                order = means[ref_col].sort_values().index.tolist()
                cluster_map = {order[i]: lvl for i, lvl in enumerate(["low", "medium", "high"][:len(order)])}
        except Exception:
            pass

        df["burnout_level"] = [cluster_map.get(c, "medium") for c in clusters]
        X = df.drop(columns=["burnout_level"]).copy()
        y = df["burnout_level"].astype(str).copy()

        logger.info(f"Cluster distribution: {pd.Series(y).value_counts().to_dict()}")



    # ---------------------
    # TRAIN FINAL PIPELINE
    # ---------------------
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    # versioning
    existing = sorted(MODELS_DIR.glob("burnout_v*.pkl"))
    version = 1 if not existing else (len(existing) + 1)
    version_file = MODELS_DIR / f"burnout_v{version}.pkl"

    joblib.dump(pipeline, version_file)
    joblib.dump(pipeline, MODEL_PATH)
    clf_pipeline = pipeline

    # Move docs: mark as trained
    for doc_id, orig in docs:
        try:
            orig["trained_at"] = datetime.utcnow().isoformat()
            orig["model_version"] = version
            db.collection("trained_surveys").add(orig)
            db.collection("untrained_surveys").document(doc_id).delete()
        except Exception:
            logger.exception(f"Failed to move doc {doc_id} to trained_surveys")

    # Save metadata to Firestore
    sample_preview = df.head(10).to_dict(orient="records")
    meta = {
        "version": version,
        "file": version_file.name,
        "created_at": datetime.utcnow().isoformat(),
        "records_used": len(df),
        "description": description,
        "source_collection": "untrained_surveys",
        "training_type": "unsupervised" if not label_cols else "supervised",
        "sample_preview": sample_preview,
    }
    db.collection("models").add(meta)

    logger.info(f"Model v{version} trained successfully using {meta['training_type']} mode.")

    return version, len(df)

# ---------------------
# 3) Retrain from DataFrame
# ---------------------
def retrain_from_dataframe(df: pd.DataFrame, description: str = "Retrained model"):
    """
    Retrain model using provided dataframe. If no 'prediction' column, will try to auto-generate pseudo-labels
    using the currently loaded model (if available). Raises on empty df.
    """
    if df.empty:
        raise ValueError("No data available for retraining")

    global clf_pipeline

    # If no 'prediction' column, attempt pseudo-label generation
    if "prediction" not in df.columns:
        logger.warning("'prediction' column missing — attempting pseudo-label generation using current model")
        if clf_pipeline is None:
            raise RuntimeError("No existing model found to generate pseudo-labels; provide labels in 'prediction' column.")
        # Determine features used by pipeline:
        feature_columns = [col for col in df.columns if col not in ["id", "created_at", "timestamp"]]
        df["prediction"] = clf_pipeline.predict(df[feature_columns])

    # Now proceed with training
    X = df.drop(columns=["prediction"]).copy()
    y = df["prediction"].astype(str).copy()

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    # versioning
    existing = sorted(MODELS_DIR.glob("burnout_v*.pkl"))
    version = 1 if not existing else (len(existing) + 1)
    version_file = MODELS_DIR / f"burnout_v{version}.pkl"

    joblib.dump(pipeline, version_file)
    joblib.dump(pipeline, MODEL_PATH)

    clf_pipeline = pipeline

    # Move untrained_surveys -> trained_surveys & annotate with version
    untrained_docs = db.collection("untrained_surveys").stream()
    moved = 0
    for doc in untrained_docs:
        try:
            data = doc.to_dict()
            data["trained_at"] = datetime.utcnow().isoformat()
            data["model_version"] = version
            db.collection("trained_surveys").add(data)
            db.collection("untrained_surveys").document(doc.id).delete()
            moved += 1
        except Exception:
            logger.exception(f"Failed to move doc {doc.id}")

    # Log model metadata
    sample_preview = df.head(10).to_dict(orient="records")
    meta = {
        "version": version,
        "file": version_file.name,
        "created_at": datetime.utcnow().isoformat(),
        "records_used": len(df),
        "description": description,
        "source_collection": "retrain_from_dataframe",
        "sample_preview": sample_preview,
    }
    db.collection("models").add(meta)

    return version, len(df)


# ---------------------
# 4) Add survey without prediction
# ---------------------
def add_survey_without_prediction(data: dict):
    db.collection("untrained_surveys").add({
        **data,
        "created_at": datetime.utcnow().isoformat(),
        "status": "untrained"
    })
    return {"message": "Survey added without prediction."}


# ---------------------
# 5) Hard reset
# ---------------------
def hard_reset():
    # delete local model files
    for p in MODELS_DIR.glob("*.pkl"):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            logger.exception(f"Failed to delete model file {p}")

    # delete Firebase data
    for col in ["models", "trained_surveys", "untrained_surveys"]:
        docs = db.collection(col).stream()
        for doc in docs:
            try:
                db.collection(col).document(doc.id).delete()
            except Exception:
                logger.exception(f"Failed to delete doc {doc.id} from {col}")

    global clf_pipeline
    clf_pipeline = None

    return {"message": "All models and Firebase data have been deleted."}


# ---------------------
# 6) Get current model info
# ---------------------
def get_current_model_info():
    if not MODEL_PATH.exists():
        return None

    # Try to get metadata doc from 'models' collection
    docs = db.collection("models").order_by("created_at", direction="DESCENDING").limit(10).stream()
    for doc in docs:
        d = doc.to_dict()
        if "file" in d:
            return {
                "version": d.get("version"),
                "file": d.get("file"),
                "created_at": d.get("created_at"),
                "records_used": d.get("records_used"),
                "description": d.get("description", "-"),
                "source_collection": d.get("source_collection"),
                "sample_preview": d.get("sample_preview", []),
            }

    # fallback to file metadata
    return {
        "version": None,
        "file": MODEL_PATH.name,
        "created_at": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
        "records_used": None,
        "description": "Unknown (no metadata)"
    }


# ---------------------
# 7) List versions from files
# ---------------------
def list_versions_from_files():
    versions = []
    for f in MODELS_DIR.glob("burnout_v*.pkl"):
        try:
            versions.append({
                "file": f.name,
                "version": int(f.stem.split("burnout_v")[-1]),
                "modified": f.stat().st_mtime
            })
        except Exception:
            logger.exception(f"Malformed model file {f}")
    return sorted(versions, key=lambda x: x["version"], reverse=True)


# ---------------------
# 8) Rollback to version
# ---------------------
def rollback_to_version(version: int):
    file = MODELS_DIR / f"burnout_v{version}.pkl"
    if not file.exists():
        raise FileNotFoundError(f"Model version {version} not found.")

    pipeline = joblib.load(file)
    joblib.dump(pipeline, MODEL_PATH)  # set as active

    global clf_pipeline
    clf_pipeline = pipeline

    # update Firebase metadata
    db.collection("models").add({
        "version": version,
        "file": file.name,
        "rolled_back_at": datetime.utcnow().isoformat(),
        "description": f"Rolled back to v{version}",
    })

    return version
