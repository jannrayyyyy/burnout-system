# train.py
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "surveys.csv"
MODEL_OUT = ROOT / "models" / "burnout_pipeline.pkl"
FEATURE_IMP_OUT = ROOT / "models" / "feature_importances.csv"
PROCESSED_OUT = ROOT / "data" / "processed_train.csv"

if not DATA_PATH.exists():
    raise SystemExit(f"Missing {DATA_PATH}. Run generate_sample_data.py or put your CSV there.")

# Edit FEATURES if your real survey has different columns:
FEATURES = [
    "age","gender","year_level","gwa","num_subjects",
    "hours_online","study_hours","sleep_hours",
    "perceived_stress","procrastination","motivation"
]
TARGET = "burnout_level"

df = pd.read_csv(DATA_PATH)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise SystemExit(f"CSV is missing columns: {missing}")

X = df[FEATURES].copy()
y = df[TARGET].astype(str).copy()

# Train/test split (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Detect numeric vs categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in FEATURES if c not in numeric_cols]

# Build OneHotEncoder with backward-compatible param
try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OHE)
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)

clf = RandomForestClassifier(random_state=42, n_jobs=-1)

pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

param_grid = {
    "clf__n_estimators": [100],
    "clf__max_depth": [None, 10],
    "clf__min_samples_split": [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1)

print("Starting training (GridSearchCV)...")
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_)

# Evaluate
y_pred = best.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save pipeline
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best, MODEL_OUT)
print(f"Saved pipeline to {MODEL_OUT}")

# Attempt to get feature names for transformed columns:
try:
    feat_names = best.named_steps['preprocessor'].get_feature_names_out()
    feat_names = list(feat_names)
except Exception:
    # fallback: numeric names + OHE feature names where possible
    feat_names = numeric_cols.copy()
    try:
        ohe_step = best.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        ohe_names = list(ohe_step.get_feature_names_out(categorical_cols))
        feat_names += ohe_names
    except Exception:
        feat_names += categorical_cols

# Save processed training data for auditing
X_train_trans = best.named_steps['preprocessor'].transform(X_train)
proc_df = pd.DataFrame(X_train_trans, columns=feat_names)
proc_df[TARGET] = y_train.reset_index(drop=True)
proc_df.to_csv(PROCESSED_OUT, index=False)
print(f"Saved processed training data to {PROCESSED_OUT}")

# Feature importances (map to feat_names if available)
try:
    importances = best.named_steps['clf'].feature_importances_
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(FEATURE_IMP_OUT, index=False)
    print(f"Saved feature importances to {FEATURE_IMP_OUT}")
except Exception as e:
    print("Could not extract feature importances:", e)

print("Training script finished.")
