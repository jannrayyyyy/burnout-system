import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "burnout_pipeline.pkl"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first.")

clf_pipeline = joblib.load(MODEL_PATH)

def run_prediction(data: dict):
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

def retrain_from_dataframe(df: pd.DataFrame):
    if "prediction" not in df.columns:
        raise ValueError("Missing 'prediction' column for retraining")

    X = df.drop(columns=["prediction"])
    y = df["prediction"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    global clf_pipeline
    clf_pipeline = clf
    return len(df)
