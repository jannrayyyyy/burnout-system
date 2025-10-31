
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


MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "burnout_pipeline.pkl"


MODELS_DIR.mkdir(parents=True, exist_ok=True)


clf_pipeline: Optional[Pipeline] = None
if MODEL_PATH.exists():
    try:
        clf_pipeline = joblib.load(MODEL_PATH)
    except Exception:
        logger.exception("Failed to load existing model file.")
        clf_pipeline = None
else:
    clf_pipeline = None





def _build_pipeline() -> Pipeline:
    """
    Build the ML pipeline. Compatible with both old and new sklearn versions.
    """
    categorical = ["gender"]

    
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        
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



def run_prediction(data: dict):
    """
    Run burnout prediction using the active model.
    Cleans input, aligns with training features, stores a consistent surveys record,
    and returns a structured explanation and probabilities.
    """
    import pandas as pd
    global clf_pipeline
    if clf_pipeline is None:
        raise RuntimeError("No model loaded. Train or deploy a model first.")

    
    MODEL_FEATURES = [
        "age", "gender", "gwa", "hours_online", "motivation",
        "num_subjects", "perceived_stress", "procrastination",
        "sleep_hours", "study_hours", "year_level"
    ]

    
    clean_data = {}
    alias_map = {
        "num_subject": "num_subjects",
        "subjects": "num_subjects",
        "hours": "hours_online",
        "perceived": "perceived_stress",
        "procrastination_level": "procrastination",
        "study": "study_hours",
        "sleep": "sleep_hours"
    }

    for k, v in data.items():
        key = alias_map.get(k.strip().lower(), k.strip().lower())
        if isinstance(v, str):
            v = v.strip().replace(";", "").lower()
        clean_data[key] = v

    
    clean_data = {k: v for k, v in clean_data.items() if k in MODEL_FEATURES}

    
    for feat in MODEL_FEATURES:
        if feat not in clean_data:
            clean_data[feat] = None

    df = pd.DataFrame([clean_data])

    
    numeric_cols = [c for c in MODEL_FEATURES if c != "gender"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    
    df[numeric_cols] = df[numeric_cols].fillna(0)
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("unknown").astype(str)

    
    try:
        proba = clf_pipeline.predict_proba(df)[0]
    except Exception as e:
        logger.exception("Prediction failed — input shape/type mismatch")
        raise RuntimeError(f"Prediction failed: {e}")
    classes = clf_pipeline.classes_
    best_idx = int(proba.argmax())
    best_class = str(classes[best_idx])
    best_prob = float(proba[best_idx])

    
    
    
    try:
        if db is not None:
            db.collection("surveys").add({
                "data": clean_data,
                "created_at": datetime.utcnow().isoformat(),
                "status": "predicted",
            })
    except Exception:
        
        logger.exception("Failed to write prediction to surveys collection")

    
    model_info = get_current_model_info() or {}
    explanation = {}
    try:
        preprocessor = clf_pipeline.named_steps["preprocessor"]
        
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            
            feature_names = []

        importances = clf_pipeline.named_steps["clf"].feature_importances_

        
        feature_importance_map = dict(zip(feature_names, importances)) if feature_names else {}
        grouped_importance = {}
        feature_display = {}

        
        if not feature_importance_map:
            for k, v in clean_data.items():
                grouped_importance[k] = 0.0
                feature_display[k] = v
        else:
            for fname, score in feature_importance_map.items():
                
                if "__" in fname:
                    base = fname.split("__")[-1]
                else:
                    base = fname

                if "_" in base and base.split("_")[0] in clean_data:
                    root = base.split("_")[0]
                    feature_key = root
                    feature_value = clean_data.get(root, base.split("_", 1)[1])
                else:
                    feature_key = base
                    feature_value = clean_data.get(feature_key, "unknown")

                grouped_importance[feature_key] = grouped_importance.get(feature_key, 0) + abs(float(score))
                feature_display[feature_key] = feature_value

        
        total = sum(grouped_importance.values()) or 1.0
        grouped_importance = {k: v / total for k, v in grouped_importance.items()}

        
        top_features = sorted(grouped_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        label_map = {
            "age": "Age",
            "gender": "Gender",
            "gwa": "GWA (General Weighted Average)",
            "hours_online": "Hours spent online per day",
            "motivation": "Motivation level",
            "num_subjects": "Number of subjects",
            "perceived_stress": "Perceived stress level",
            "procrastination": "Procrastination level",
            "sleep_hours": "Sleep hours per day",
            "study_hours": "Study hours per day",
            "year_level": "Year level"
        }

        def human_label(name: str):
            return label_map.get(name, name.replace("_", " ").capitalize())

        summary_parts = []
        for feature, importance in top_features:
            label = human_label(feature)
            val = feature_display.get(feature, "unspecified")
            if pd.isna(val) or val in ["unknown", None, "nan"]:
                val = "unspecified"

            if importance > 0.15:
                summary_parts.append(f"{label} ({val}) shows a strong influence on burnout risk.")
            elif importance > 0.05:
                summary_parts.append(f"{label} ({val}) moderately affects the predicted burnout level.")
            else:
                summary_parts.append(f"{label} ({val}) has a minor impact on the burnout prediction.")

        explanation = {
            "top_features": {k: feature_display.get(k, "unspecified") for k, _ in top_features},
            "feature_contributions": {k: round(v, 3) for k, v in grouped_importance.items()},
            "summary": (" ".join(summary_parts) if summary_parts else "No dominant features were identified for this prediction.")
        }

    except Exception as e:
        explanation = {"summary": f"Explanation unavailable: {str(e)}"}

    
    return {
        "burnout_level": best_class,
        "probability": best_prob,
        "all_probabilities": dict(zip(map(str, classes.tolist()), proba.tolist())),
        "model_version": model_info.get("version"),
        "model_records_used": model_info.get("records_used"),
        "model_created_at": model_info.get("created_at"),
        "explanation": explanation,
    }


def retrain_from_dataframe(df: pd.DataFrame, description: str = "Retrained model"):
    """
    Retrain model using provided dataframe.
    If 'prediction' is missing, pseudo-label using current model.
    Ensures consistent preprocessing schema to prevent NaN predictions.
    """
    if df.empty:
        raise ValueError("No data available for retraining")

    global clf_pipeline
    if clf_pipeline is None and "prediction" not in df.columns:
        raise RuntimeError("No existing model found to generate pseudo-labels")

    
    if "prediction" not in df.columns:
        logger.warning("No 'prediction' column found — generating pseudo-labels using current model")
        feature_columns = [
            col for col in df.columns
            if col not in ["id", "created_at", "timestamp", "status", "trained_at", "model_version"]
        ]

        df_pred = df.copy()
        for col in feature_columns:
            if df_pred[col].dtype == "object":
                df_pred[col] = df_pred[col].astype(str)

        
        df["prediction"] = clf_pipeline.predict(df_pred[feature_columns])

    
    MODEL_FEATURES = [
        "age", "gender", "gwa", "hours_online", "motivation",
        "num_subjects", "perceived_stress", "procrastination",
        "sleep_hours", "study_hours", "year_level"
    ]

    
    alias_map = {
        "num_subject": "num_subjects",
        "subjects": "num_subjects",
        "hours": "hours_online",
        "perceived": "perceived_stress",
        "procrastination_level": "procrastination",
        "study": "study_hours",
        "sleep": "sleep_hours"
    }

    clean_records = []
    for _, row in df.iterrows():
        clean = {}
        for k, v in row.items():
            key = alias_map.get(str(k).strip().lower(), str(k).strip().lower())
            clean[key] = v
        clean_records.append(clean)

    clean_df = pd.DataFrame(clean_records)

    
    label_col = "prediction"
    for feat in MODEL_FEATURES:
        if feat not in clean_df.columns:
            clean_df[feat] = np.nan

    X = clean_df[MODEL_FEATURES].copy()
    y = clean_df[label_col].astype(str).copy()

    
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    
    existing = sorted(MODELS_DIR.glob("burnout_v*.pkl"))
    version = len(existing) + 1
    version_file = MODELS_DIR / f"burnout_v{version}.pkl"

    joblib.dump(pipeline, version_file)
    joblib.dump(pipeline, MODEL_PATH)
    clf_pipeline = pipeline

    
    sample_preview = clean_df.head(10).to_dict(orient="records")
    meta = {
        "version": version,
        "file": version_file.name,
        "created_at": datetime.utcnow().isoformat(),
        "records_used": len(clean_df),
        "description": description,
        "source_collection": "retrain_from_dataframe",
        "training_type": "retrain",
        "sample_preview": sample_preview,
        "active": True,
        "activated_at": datetime.utcnow().isoformat()
    }

    if db is not None:
        db.collection("models").add(meta)

    logger.info(f"Model v{version} retrained successfully on {len(clean_df)} records.")
    return version, len(clean_df)


def add_survey_without_prediction(data: dict):
    if db is not None:
        db.collection("surveys").add({
            "data": data,
            "created_at": datetime.utcnow().isoformat(),
            "status": "untrained"
        })
        return {"message": "Survey added without prediction."}
    else:
        
        return {"message": "Firebase not configured; survey received locally."}


def hard_reset():
    
    for p in MODELS_DIR.glob("*.pkl"):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            logger.exception(f"Failed to delete model file {p}")

    
    
    if db is not None:
        for col in ["models", "surveys"]:
            docs = db.collection(col).stream()
            for doc in docs:
                try:
                    db.collection(col).document(doc.id).delete()
                except Exception:
                    logger.exception(f"Failed to delete doc {doc.id} from {col}")

    global clf_pipeline
    clf_pipeline = None

    return {"message": "All models and Firebase data have been deleted."}


def get_current_model_info():
    if not MODEL_PATH.exists():
        return None

    try:
        
        docs = db.collection("models").order_by("trained_at", direction="DESCENDING").limit(20).stream()
        latest_doc = None
        active_doc = None

        for doc in docs:
            d = doc.to_dict() or {}
            d["id"] = doc.id

            
            if latest_doc is None:
                latest_doc = d

        model_doc = active_doc or latest_doc
        if model_doc:
            return {
                "version": model_doc.get("version"),
                "file": model_doc.get("file"),
                "created_at": model_doc.get("created_at"),
                "records_used": model_doc.get("records_used"),
                "description": model_doc.get("description", "-"),
                "source_collection": model_doc.get("source_collection", "-"),
                "training_type": model_doc.get("training_type", "-"),
                
                "active": model_doc.get("active", False),
            }

    except Exception as e:
        import logging
        logging.exception("Failed to fetch model info from Firestore: %s", e)

    
    return {
        "version": None,
        "file": MODEL_PATH.name,
        "created_at": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
        "records_used": None,
        "description": "Active local model (no metadata)",
        "active": True,
    }





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





def rollback_to_version(version: int):
    file = MODELS_DIR / f"burnout_v{version}.pkl"
    if not file.exists():
        raise FileNotFoundError(f"Model version {version} not found.")

    pipeline = joblib.load(file)
    joblib.dump(pipeline, MODEL_PATH)  

    global clf_pipeline
    clf_pipeline = pipeline

    
    db.collection("models").add({
        "version": version,
        "file": file.name,
        "rolled_back_at": datetime.utcnow().isoformat(),
        "description": f"Rolled back to v{version}",
        "active": True,
        "activated_at": datetime.utcnow().isoformat()
    })

    return version
