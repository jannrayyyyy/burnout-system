# generate_sample_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys, os
import joblib

# Make sure backend modules are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))
from services.firebase_service import db
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def generate_sample_data(n: int = 100):
    """Generate synthetic burnout survey data and upload metadata to Firestore."""
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    np.random.seed(42)

    # -----------------------------
    # 1️⃣ Generate Dataset
    # -----------------------------
    df = pd.DataFrame({
        "age": np.random.randint(17, 25, n),
        "gender": np.random.choice(["male", "female", "other"], n, p=[0.45, 0.45, 0.10]),
        "year_level": np.random.randint(1, 6, n),
        "gwa": np.round(np.random.uniform(1.0, 5.0, n), 2),
        "num_subjects": np.random.randint(3, 10, n),
        "hours_online": np.round(np.random.uniform(0, 8, n), 1),
        "study_hours": np.round(np.random.uniform(0, 12, n), 1),
        "sleep_hours": np.round(np.random.uniform(3, 9, n), 1),
        "perceived_stress": np.random.randint(1, 6, n),
        "procrastination": np.random.randint(1, 6, n),
        "motivation": np.random.randint(1, 6, n),
    })

    # Simple burnout formula for demo
    sleep_penalty = pd.cut(df["sleep_hours"], bins=[-1, 4, 6, 100], labels=[2, 1, 0]).astype(int)
    score = df["perceived_stress"] + (6 - df["motivation"]) + sleep_penalty
    df["burnout_level"] = pd.cut(score, bins=[-1, 3, 6, 100], labels=["Low", "Moderate", "High"])

    output_path = Path("data/surveys.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Wrote {output_path} with {len(df)} rows")

    # -----------------------------
    # 2️⃣ Train a Model
    # -----------------------------
    X = df.drop(columns=["burnout_level"])
    y = df["burnout_level"]

    categorical = ["gender"]
    numeric = [col for col in X.columns if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)

    # -----------------------------
    # 3️⃣ Save Model
    # -----------------------------
    model_version = "test"
    model_path = Path(f"models/model_{model_version}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"✅ Trained model saved to {model_path} (accuracy: {acc:.2f})")

    # -----------------------------
    # 4️⃣ Upload Metadata to Firebase
    # -----------------------------

    metadata = {
        "version": model_version,
        "description": "Sample burnout model trained on synthetic dataset",
        "created_at": datetime.utcnow().isoformat(),
        "records_used": len(df),
        "file": str(model_path),
        "source_collection": "synthetic_data",
        "active": True,
        "accuracy": round(acc, 3),
    }

    # Upload to Firestore
    db.collection("models").add(metadata)
    print("✅ Uploaded model metadata to Firestore (models collection)")
    print(json.dumps(metadata, indent=2))

    return metadata


if __name__ == "__main__":
    print("🚀 Generating sample burnout dataset and training model...")
    result = generate_sample_data(n=100)
    print("🎯 Done — model and data successfully generated!\n")
