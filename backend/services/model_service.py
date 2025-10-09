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
    """
    Run burnout prediction using the active model.
    Cleans input, aligns with training features, stores a consistent surveys record,
    and returns a structured explanation and probabilities.
    """
    import pandas as pd
    global clf_pipeline
    if clf_pipeline is None:
        raise RuntimeError("No model loaded. Train or deploy a model first.")

    # Step 1: Define valid features (must match training)
    MODEL_FEATURES = [
        "age", "gender", "gwa", "hours_online", "motivation",
        "num_subjects", "perceived_stress", "procrastination",
        "sleep_hours", "study_hours", "year_level"
    ]

    # Step 2: Normalize and clean incoming data
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

    # Keep only valid model features
    clean_data = {k: v for k, v in clean_data.items() if k in MODEL_FEATURES}

    # Step 3: Ensure all MODEL_FEATURES exist and coerce types to avoid NaNs
    for feat in MODEL_FEATURES:
        if feat not in clean_data:
            clean_data[feat] = None

    df = pd.DataFrame([clean_data])

    # Coerce numeric-like columns to numeric (except gender)
    numeric_cols = [c for c in MODEL_FEATURES if c != "gender"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric values with 0 (safe default) and gender with 'unknown'
    df[numeric_cols] = df[numeric_cols].fillna(0)
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("unknown").astype(str)

    # Step 4: Run prediction
    try:
        proba = clf_pipeline.predict_proba(df)[0]
    except Exception as e:
        logger.exception("Prediction failed — input shape/type mismatch")
        raise RuntimeError(f"Prediction failed: {e}")
    classes = clf_pipeline.classes_
    best_idx = int(proba.argmax())
    best_class = str(classes[best_idx])
    best_prob = float(proba[best_idx])

    # Step 5: Log the prediction for retraining later.
    # Save a consistent document shape: keep raw survey under `data` and attach prediction metadata.
    # Persist prediction into unified 'surveys' collection when possible
    try:
        if db is not None:
            db.collection("surveys").add({
                "data": clean_data,
                "created_at": datetime.utcnow().isoformat(),
                "status": "predicted",
            })
    except Exception:
        # Don't fail prediction if logging to Firestore fails; just warn.
        logger.exception("Failed to write prediction to surveys collection")

    # Step 6: Model explainability
    model_info = get_current_model_info() or {}
    explanation = {}
    try:
        preprocessor = clf_pipeline.named_steps["preprocessor"]
        # sklearn's ColumnTransformer + OneHotEncoder may expose get_feature_names_out
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            # Fallback: synthesize feature names from pipeline components
            feature_names = []

        importances = clf_pipeline.named_steps["clf"].feature_importances_

        # Map encoded feature names back to human features
        feature_importance_map = dict(zip(feature_names, importances)) if feature_names else {}
        grouped_importance = {}
        feature_display = {}

        # If we don't have encoder feature names, use raw features from clean_data
        if not feature_importance_map:
            for k, v in clean_data.items():
                grouped_importance[k] = 0.0
                feature_display[k] = v
        else:
            for fname, score in feature_importance_map.items():
                # attempt to recover base feature name
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

        # Normalize importances
        total = sum(grouped_importance.values()) or 1.0
        grouped_importance = {k: v / total for k, v in grouped_importance.items()}

        # Select top 5
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

    # Step 7: Return the full structured result
    return {
        "burnout_level": best_class,
        "probability": best_prob,
        "all_probabilities": dict(zip(map(str, classes.tolist()), proba.tolist())),
        "model_version": model_info.get("version"),
        "model_records_used": model_info.get("records_used"),
        "model_created_at": model_info.get("created_at"),
        "explanation": explanation,
    }

# ---------------------
# 2) Train from untrained_surveys (one-shot)
# ---------------------
def train_from_untrained(description: str = "Initial trained model") -> Tuple[int, int]:
    """
    Train a model using data from the `surveys` collection.
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

        # Robust feature construction:
        # - Try to coerce numeric-like columns to numeric
        # - Encode low-cardinality categorical columns (e.g., gender, year_level)
        candidate_features = [
            "age", "gender", "gwa", "hours_online", "motivation",
            "num_subjects", "perceived_stress", "procrastination",
            "sleep_hours", "study_hours", "year_level"
        ]

        X_work = df.copy()

        # Try to coerce possible numeric columns
        for col in list(X_work.columns):
            # Skip obviously non-feature columns
            if col in ["id"]:
                continue
            # If column is object, try converting to numeric
            if X_work[col].dtype == object:
                coerced = pd.to_numeric(X_work[col], errors="coerce")
                # If at least one non-na conversion succeeded, use numeric
                if coerced.notna().any():
                    X_work[col] = coerced

        # Build features list: numeric columns plus encoded low-cardinality categoricals
        numeric_df = X_work.select_dtypes(include=[np.number]).copy()

        # Label-encode low-cardinality categoricals (<=10 unique non-null values)
        categorical_cols = []
        for col in X_work.select_dtypes(include=[object]).columns:
            nunique = X_work[col].nunique(dropna=True)
            if 0 < nunique <= 10 and col in candidate_features:
                # factorize into numeric codes; reserve -1 for NaN
                codes, uniques = pd.factorize(X_work[col].fillna("__MISSING__"))
                categorical_cols.append((col, codes))

        # Combine numeric and encoded categorical features
        feature_df = numeric_df.copy()
        for col, codes in categorical_cols:
            feature_df[col] = codes

        # Drop all-NaN columns
        feature_df = feature_df.dropna(axis=1, how="all")

        # Fill numeric NaNs with median
        from sklearn.impute import SimpleImputer
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            feature_df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(feature_df[numeric_cols])

        # Drop constant columns
        feature_df = feature_df.loc[:, feature_df.apply(pd.Series.nunique) > 1]
        if feature_df.empty:
            # Provide a helpful error including available columns and sample values
            sample = df.head(5).to_dict(orient="records")
            raise ValueError(
                "Cannot perform unsupervised training: no usable feature columns found (all constant or non-numeric). "
                "Include numeric survey fields like 'age', 'hours_online', 'gwa', or add variability to the dataset. "
                f"Sample rows: {sample}"
            )

        # Determine cluster count
        import numpy as np
        from sklearn.cluster import KMeans
        unique_rows = np.unique(feature_df.to_numpy(), axis=0)
        n_clusters = min(3, len(unique_rows))
        if n_clusters < 2:
            raise ValueError("Not enough unique data rows to form clusters. Add more diverse survey responses.")

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

    # Mark documents as trained in the unified 'surveys' collection
    for doc_id, orig in docs:
        try:
            if db is not None:
                db.collection("surveys").document(doc_id).update({
                    "trained_at": datetime.utcnow().isoformat(),
                    "model_version": version,
                    "status": "trained",
                })
        except Exception:
            logger.exception(f"Failed to mark doc {doc_id} as trained")

    # Save metadata to Firestore
    sample_preview = df.head(10).to_dict(orient="records")
    meta = {
        "version": version,
        "file": version_file.name,
        "created_at": datetime.utcnow().isoformat(),
        "records_used": len(df),
        "description": description,
        "source_collection": "surveys",
        "training_type": "unsupervised" if not label_cols else "supervised",
        "sample_preview": sample_preview,
        "active": True,
        "activated_at": datetime.utcnow().isoformat()
    }
    db.collection("models").add(meta)

    logger.info(f"Model v{version} trained successfully using {meta['training_type']} mode.")

    return version, len(df)

# ---------------------
# 3) Retrain from DataFrame
# ---------------------
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

    # --- Generate pseudo-labels if needed ---
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

        # Use existing model to predict pseudo-labels
        df["prediction"] = clf_pipeline.predict(df_pred[feature_columns])

    # --- Align features to the training schema ---
    MODEL_FEATURES = [
        "age", "gender", "gwa", "hours_online", "motivation",
        "num_subjects", "perceived_stress", "procrastination",
        "sleep_hours", "study_hours", "year_level"
    ]

    # Normalize field names
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

    # Keep only model features + label
    label_col = "prediction"
    for feat in MODEL_FEATURES:
        if feat not in clean_df.columns:
            clean_df[feat] = np.nan

    X = clean_df[MODEL_FEATURES].copy()
    y = clean_df[label_col].astype(str).copy()

    # --- Train new pipeline using same preprocessing ---
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    # --- Save new version ---
    existing = sorted(MODELS_DIR.glob("burnout_v*.pkl"))
    version = len(existing) + 1
    version_file = MODELS_DIR / f"burnout_v{version}.pkl"

    joblib.dump(pipeline, version_file)
    joblib.dump(pipeline, MODEL_PATH)
    clf_pipeline = pipeline

    # --- Log metadata ---
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


# ---------------------
# 4) Add survey without prediction
# ---------------------
def add_survey_without_prediction(data: dict):
    if db is not None:
        db.collection("surveys").add({
            "data": data,
            "created_at": datetime.utcnow().isoformat(),
            "status": "untrained"
        })
        return {"message": "Survey added without prediction."}
    else:
        # fallback to local acknowledgement when Firebase is disabled
        return {"message": "Firebase not configured; survey received locally."}


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
    # delete Firebase data
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


# ---------------------
# 6) Get current model info
# ---------------------
def get_current_model_info():
    if not MODEL_PATH.exists():
        return None

    try:
        # Fetch models ordered by created_at descending
        docs = db.collection("models").order_by("activated_at", direction="DESCENDING").limit(20).stream()
        latest_doc = None
        active_doc = None

        for doc in docs:
            d = doc.to_dict() or {}
            d["id"] = doc.id

            # Prefer the one explicitly marked as active
            if d.get("active") is True:
                active_doc = d
                break

            # Fallback: remember the most recent
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
                # "sample_preview": model_doc.get("sample_preview", []),
                "active": model_doc.get("active", False),
            }

    except Exception as e:
        import logging
        logging.exception("Failed to fetch model info from Firestore: %s", e)

    # Fallback: use the active file on disk
    return {
        "version": None,
        "file": MODEL_PATH.name,
        "created_at": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
        "records_used": None,
        "description": "Active local model (no metadata)",
        "active": True,
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
        "active": True,
        "activated_at": datetime.utcnow().isoformat()
    })

    return version
