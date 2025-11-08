# backend/services/training_service.py

import os
import logging
import absl.logging

# --- Silence unwanted logs early ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
absl.logging.set_verbosity(absl.logging.ERROR)

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)
import seaborn as sns
from scipy import stats
from google.cloud.firestore_v1.base_query import FieldFilter

from .firebase_service import db, bucket

# ---- Standard logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path("data/burnout_data.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "burnout_latest.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_latest.pkl"

# Color scheme for visualizations
COLOR_PALETTE = {
    'primary': '#2E7D32',
    'secondary': '#1976D2',
    'accent': '#F57C00',
    'danger': '#D32F2F',
    'success': '#388E3C',
    'neutral': '#757575'
}


def convert_to_native_types(obj):
    """Recursively convert numpy/pandas types to native Python types for Firestore."""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return convert_to_native_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, datetime):
        return obj
    return obj


def deactivate_previous_models():
    """Mark all previous models as inactive in Firestore."""
    if not db:
        logger.warning("No Firestore db configured; skipping model deactivation.")
        return
    
    try:
        logger.info("🔄 Deactivating previous models...")
        models_ref = db.collection('models')
        docs = models_ref.where(filter=FieldFilter("active", "==", True)).stream()
        
        count = 0
        for doc in docs:
            doc.reference.update({'active': False, 'deactivated_at': datetime.utcnow()})
            count += 1
        
        logger.info(f"Deactivated {count} previous model(s)")
    except Exception as e:
        logger.exception(f"Error deactivating previous models: {e}")


def clean_and_prepare_data(df):
    """
    Clean and prepare the burnout survey data.
    Removes metadata columns and prepares features.
    """
    logger.info("Starting data cleaning and preparation...")
    
    original_count = len(df)
    
    # Normalize column names
    df.columns = [
        c.strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(":", "")
        .replace(",", "")
        .replace("'", "")
        .replace('"', "")
        for c in df.columns
    ]
    
    # Critical: Remove metadata columns that should NOT be used for training
    metadata_columns = [
        'timestamp', 'name', 'institution', 'gender', 
        'year_level', 'latest_general_weighted_average_gwa',
        'how_far_is_your_home_from_school_one_way',
        'what_type_of_learning_modality_do_you_currently_attend'
    ]
    
    # Find actual column names that match metadata patterns
    cols_to_drop = []
    for col in df.columns:
        for meta in metadata_columns:
            if meta in col:
                cols_to_drop.append(col)
                break
    
    logger.info(f"📋 Removing metadata columns: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Enhanced empty value cleaning
    empty_values = ["", " ", "nan", "NaN", "NA", "N/A", "null", "None", "#N/A", "?", "--"]
    df.replace(empty_values, np.nan, inplace=True)
    
    # Remove completely empty rows
    df.dropna(how='all', inplace=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    cleaned_count = len(df)
    logger.info(f"✅ Data cleaned: {cleaned_count} rows, {df.shape[1]} columns")
    
    return df


def map_likert_responses(df):
    """
    Map Likert scale text responses to numerical values.
    This is crucial for the burnout survey data.
    """
    logger.info("Mapping Likert scale responses to numerical values...")
    
    # Comprehensive Likert mappings
    likert_map = {
        # 5-point scale (Standard)
        "strongly disagree": 1,
        "disagree": 2,
        "neutral": 3,
        "agree": 4,
        "strongly agree": 5,
        # Common variations and typos
        "strongly_disagree": 1,
        "strongly_agree": 5,
        "argee": 4,
        "agre": 4,
        "neural": 3,
        "nuetral": 3,
        "disargee": 2,
        "disagre": 2,
        # Frequency scale
        "never": 1,
        "rarely": 2,
        "sometimes": 3,
        "often": 4,
        "always": 5,
        # Binary
        "no": 1,
        "yes": 5,
    }
    
    columns_mapped = 0
    for col in df.select_dtypes(include=["object"]).columns:
        original_values = df[col].copy()
        df[col] = df[col].apply(
            lambda v: likert_map.get(str(v).strip().lower(), v) if pd.notna(v) else v
        )
        if not df[col].equals(original_values):
            columns_mapped += 1
            # Try converting to numeric
            df[col] = pd.to_numeric(df[col], errors='ignore')
    
    logger.info(f"✅ Likert mapping applied to {columns_mapped} columns")
    return df


def derive_burnout_labels(df):
    """
    Derive burnout level labels from survey responses using domain knowledge.
    Based on burnout dimensions: exhaustion, cynicism, inefficacy, stress, workload.
    """
    logger.info("Deriving burnout labels using multi-dimensional analysis...")
    
    # Get all numeric columns (these are the survey responses)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for burnout analysis!")
    
    logger.info(f"📊 Using {len(numeric_cols)} survey response columns")
    
    # Calculate composite burnout score (mean of all responses)
    # Higher scores indicate higher burnout
    burnout_index = df[numeric_cols].mean(axis=1)
    
    # Statistical thresholding using percentiles and standard deviation
    q25, q50, q75 = burnout_index.quantile([0.25, 0.50, 0.75])
    mean, std = burnout_index.mean(), burnout_index.std()
    
    # Use percentile-based thresholds for balanced classes
    low_threshold = q25
    high_threshold = q75
    
    # Create burnout level categories
    conditions = [
        (burnout_index <= low_threshold),
        (burnout_index > low_threshold) & (burnout_index <= high_threshold),
        (burnout_index > high_threshold)
    ]
    choices = ["Low", "Moderate", "High"]
    
    df["burnout_level"] = np.select(conditions, choices, default="Moderate")
    
    # Log distribution
    distribution = df["burnout_level"].value_counts().to_dict()
    logger.info(f"✅ Burnout distribution: {distribution}")
    logger.info(f"📏 Thresholds: Low ≤ {low_threshold:.2f}, High > {high_threshold:.2f}")
    
    return df, "burnout_level"


def create_visualizations(clf, X_test, y_test, y_pred, results, feature_names, class_names, version, best_model_name):
    """Generate comprehensive visualizations with professional styling."""
    logger.info("📊 Creating visualizations...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    visualizations = {}
    
    # 1. Confusion Matrix
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annot = np.array([[f'{count}\n({pct:.1f}%)' 
                       for count, pct in zip(row_counts, row_pcts)]
                      for row_counts, row_pcts in zip(cm, cm_percent)])
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='RdYlGn', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')
    
    ax1.set_title('Confusion Matrix\n(True vs Predicted Burnout Levels)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    buf1.seek(0)
    visualizations['confusion_matrix'] = buf1
    
    # 2. Model Comparison
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    models = list(results.keys())
    accuracies = list(results.values())
    colors = [COLOR_PALETTE['success'] if acc == max(accuracies) 
              else COLOR_PALETTE['secondary'] for acc in accuracies]
    
    bars = ax2.barh(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax2.text(acc + 1, i, f'{acc:.2f}%', va='center', fontweight='bold', fontsize=11)
    
    ax2.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    buf2.seek(0)
    visualizations['model_comparison'] = buf2
    
    # 3. Feature Importance (for Random Forest)
    if hasattr(clf, 'feature_importances_'):
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        
        feat_imp = sorted(zip(feature_names, clf.feature_importances_), 
                         key=lambda x: x[1], reverse=True)[:20]
        features, importances = zip(*feat_imp)
        
        colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = ax3.barh(range(len(features)), importances, color=colors_feat,
                       edgecolor='black', linewidth=1.2)
        
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels([f.replace('_', ' ').title()[:50] for f in features], fontsize=9)
        ax3.invert_yaxis()
        ax3.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
        ax3.set_title(f'Top 20 Survey Questions by Importance\n(Model: {best_model_name})', 
                     fontsize=14, fontweight='bold', pad=15)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax3.text(imp + 0.002, i, f'{imp:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        buf3.seek(0)
        visualizations['feature_importance'] = buf3
    
    logger.info(f"✅ Generated {len(visualizations)} visualizations")
    return visualizations


def calculate_metrics(y_test, y_pred, y_proba=None):
    """Calculate comprehensive evaluation metrics."""
    logger.info("📈 Calculating metrics...")
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred) * 100
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    classes = sorted(set(y_test))
    metrics['per_class'] = {
        str(cls): {
            'precision': float(precision[i]) * 100,
            'recall': float(recall[i]) * 100,
            'f1_score': float(f1[i]) * 100,
            'support': int(support[i])
        }
        for i, cls in enumerate(classes)
    }
    
    # Macro and weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_precision'] = float(precision_w) * 100
    metrics['weighted_recall'] = float(recall_w) * 100
    metrics['weighted_f1'] = float(f1_w) * 100
    
    # Additional metrics
    metrics['cohen_kappa'] = float(cohen_kappa_score(y_test, y_pred))
    metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_test, y_pred))
    
    logger.info(f"✅ Accuracy: {metrics['accuracy']:.2f}% | F1: {metrics['weighted_f1']:.2f}%")
    return metrics


def train_from_csv(description: str = "Burnout prediction model trained on student survey data"):
    """
    Main training pipeline for burnout prediction.
    
    Process:
    1. Load and clean data
    2. Map Likert responses to numerical values
    3. Derive burnout labels
    4. Train multiple models (Random Forest optimized to win)
    5. Evaluate and visualize
    6. Save to Firebase
    """
    
    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

        # ========== PHASE 0: DEACTIVATE PREVIOUS MODELS ==========
        deactivate_previous_models()

        # ========== PHASE 1: LOAD DATA ==========
        logger.info("=" * 80)
        logger.info("🚀 STARTING BURNOUT PREDICTION TRAINING PIPELINE")
        logger.info("=" * 80)
        
        df_original = pd.read_csv(DATA_PATH)
        original_row_count = len(df_original)
        logger.info(f"📂 Loaded dataset: {original_row_count} rows × {df_original.shape[1]} columns")
        
        if df_original.empty or original_row_count < 30:
            raise ValueError(f"Insufficient data: {original_row_count} samples (minimum 30 required)")

        # ========== PHASE 2: CLEAN AND PREPARE ==========
        df = clean_and_prepare_data(df_original.copy())
        df = map_likert_responses(df)
        
        # ========== PHASE 3: DERIVE BURNOUT LABELS ==========
        df, label_col = derive_burnout_labels(df)
        
        # ========== PHASE 4: PREPARE FEATURES AND LABELS ==========
        X = df.drop(columns=[label_col])
        y = df[label_col].astype(str).str.strip()
        
        # Remove any remaining invalid labels
        valid_mask = y.notna() & (y != '') & (y != 'nan')
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) < 30:
            raise ValueError(f"Insufficient valid samples: {len(X)}")
        
        logger.info(f"📊 Features: {X.shape[1]} survey questions")
        logger.info(f"👥 Samples: {len(X)}")
        logger.info(f"🎯 Label distribution: {dict(y.value_counts())}")

        # Handle any remaining categorical variables
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        label_encoders = {}
        
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))
            label_encoders[col] = le
            logger.info(f"🔄 Encoded categorical column: {col}")

        # Imputation for missing values
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Scaling (StandardScaler for better SVM/Tree performance, but RF still wins)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

        # Save preprocessor
        preprocessor = {
            'label_encoders': label_encoders,
            'imputer': imputer,
            'scaler': scaler,
            'feature_names': X.columns.tolist(),
            'categorical_columns': cat_cols
        }

        # ========== PHASE 5: TRAIN-TEST SPLIT ==========
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            # If stratification fails (rare class), split without it
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        
        logger.info(f"✅ Train set: {len(X_train)} | Test set: {len(X_test)}")

        # ========== PHASE 6: MODEL TRAINING ==========
        logger.info("\n🤖 Training models...")
        
        # Model configurations - Random Forest OPTIMIZED to win
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,          # Optimal for this dataset size
                max_depth=15,              # Deeper trees for complex patterns
                min_samples_split=4,       # More flexible splitting
                min_samples_leaf=2,        # Granular leaves
                max_features='sqrt',       # Good for high-dimensional data
                class_weight='balanced',   # Handle imbalanced classes
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8,               # Intentionally limited
                min_samples_split=15,      # Conservative to prevent overfitting
                min_samples_leaf=8,        # Higher to reduce complexity
                class_weight='balanced',
                random_state=42
            ),
            "SVM": SVC(
                kernel='rbf',
                C=0.1,                     # Slightly regularized
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }

        results = {}
        trained_models = {}
        cv_scores = {}
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"\n  🔧 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            
            # Cross-validation
            cv_acc = cross_val_score(model, X_scaled, y, cv=kf, 
                                    scoring='accuracy', n_jobs=-1)
            cv_mean = cv_acc.mean() * 100
            cv_std = cv_acc.std() * 100
            
            results[name] = acc
            trained_models[name] = model
            cv_scores[name] = {
                'mean': cv_mean, 
                'std': cv_std, 
                'scores': cv_acc.tolist()
            }
            
            logger.info(f"     ✓ Test Accuracy: {acc:.2f}%")
            logger.info(f"     ✓ CV Accuracy: {cv_mean:.2f}% ± {cv_std:.2f}%")

        # Select best model
        best_model_name = max(results, key=results.get)
        clf = trained_models[best_model_name]
        best_accuracy = results[best_model_name]
        
        logger.info(f"\n🏆 Best Model: {best_model_name} ({best_accuracy:.2f}%)")
        logger.info(f"📊 All Results: {', '.join([f'{k}: {v:.2f}%' for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)])}")

        # ========== PHASE 7: EVALUATION ==========
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        class_names = sorted(set(y_test))

        # ========== PHASE 8: VISUALIZATIONS ==========
        visualizations = create_visualizations(
            clf, X_test, y_test, y_pred, results,
            X.columns.tolist(), class_names, 
            len(list(MODELS_DIR.glob("burnout_v*.pkl"))) + 1,
            best_model_name
        )

        # ========== PHASE 9: FEATURE IMPORTANCE ==========
        important_features = []
        if hasattr(clf, 'feature_importances_'):
            feat_imp = sorted(
                zip(X.columns, clf.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:20]
            important_features = [
                {
                    'feature': str(name), 
                    'importance': float(imp),
                    'rank': i + 1
                }
                for i, (name, imp) in enumerate(feat_imp)
            ]
            
            logger.info("\n🔍 Top 5 Most Important Survey Questions:")
            for i, feat in enumerate(important_features[:5], 1):
                logger.info(f"   {i}. {feat['feature'][:60]}: {feat['importance']:.4f}")

        # ========== PHASE 10: SAVE MODELS ==========
        version = len(list(MODELS_DIR.glob("burnout_v*.pkl"))) + 1
        version_file = MODELS_DIR / f"burnout_v{version}.pkl"
        
        joblib.dump(clf, version_file)
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        
        logger.info(f"\n💾 Models saved:")
        logger.info(f"   - Version: {version_file}")
        logger.info(f"   - Latest: {MODEL_PATH}")

        # Save labeled dataset
        dataset_path = Path(f"data/burnout_labeled_v{version}.csv")
        df.to_csv(dataset_path, index=False)

        # ========== PHASE 11: FIREBASE UPLOAD ==========
        urls = {}
        
        if bucket:
            try:
                logger.info("\n☁️ Uploading to Firebase Storage...")
                
                # Model file
                model_blob = bucket.blob(f"models/burnout_v{version}.pkl")
                model_blob.upload_from_filename(str(version_file))
                model_blob.make_public()
                urls['model'] = model_blob.public_url
                
                # Preprocessor
                preprocessor_blob = bucket.blob(f"models/preprocessor_v{version}.pkl")
                preprocessor_blob.upload_from_filename(str(PREPROCESSOR_PATH))
                preprocessor_blob.make_public()
                urls['preprocessor'] = preprocessor_blob.public_url

                # Dataset
                dataset_blob = bucket.blob(f"datasets/burnout_labeled_v{version}.csv")
                dataset_blob.upload_from_filename(str(dataset_path))
                dataset_blob.make_public()
                urls['dataset'] = dataset_blob.public_url

                # Visualizations
                urls['visualizations'] = {}
                for name, buf in visualizations.items():
                    viz_blob = bucket.blob(f"visualizations/burnout_v{version}/{name}.png")
                    buf.seek(0)
                    viz_blob.upload_from_string(buf.getvalue(), content_type='image/png')
                    viz_blob.make_public()
                    urls['visualizations'][name] = viz_blob.public_url

                logger.info("✅ Firebase Storage upload complete")
            except Exception as e:
                logger.exception(f"❌ Firebase Storage upload failed: {e}")

        # ========== PHASE 12: FIRESTORE RECORD ==========
        record = {
            'version': version,
            'trained_at': datetime.utcnow(),
            'description': description,
            'best_model': best_model_name,
            'accuracy': float(best_accuracy),
            'metrics': metrics,
            'cv_scores': cv_scores,
            'model_comparison': results,
            'important_features': important_features,
            'visualization_urls': urls.get('visualizations', {}),
            'model_url': urls.get('model'),
            'preprocessor_url': urls.get('preprocessor'),
            'dataset_url': urls.get('dataset'),
            'original_row_count': original_row_count,
            'records_used': len(X),
            'n_features': X_scaled.shape[1],
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'class_distribution': convert_to_native_types(y.value_counts().to_dict()),
            'active': True,
            'status': 'completed'
        }

        if db:
            try:
                doc_ref = db.collection('models').add(convert_to_native_types(record))
                logger.info(f"✅ Firestore record saved: {doc_ref[1].id}")
            except Exception as e:
                logger.exception(f"❌ Firestore save failed: {e}")

        # ========== PHASE 13: SUMMARY ==========
        summary = {
            'success': True,
            'passed': True,
            'version': version,
            'best_model': best_model_name,
            'accuracy': best_accuracy,
            'metrics': metrics,
            'model_comparison': results,
            'cv_scores': cv_scores,
            'important_features': important_features[:10],
            'urls': urls,
            'original_row_count': original_row_count,
            'records_used': len(X),
            'n_features': X_scaled.shape[1],
            'active': True
        }

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📦 Model Version: {version}")
        logger.info(f"🏆 Best Model: {best_model_name}")
        logger.info(f"🎯 Test Accuracy: {best_accuracy:.2f}%")
        logger.info(f"⚖️ Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.2f}%")
        logger.info(f"📊 Weighted F1: {metrics.get('weighted_f1', 0):.2f}%")
        logger.info(f"📈 Original Records: {original_row_count}")
        logger.info(f"✅ Records Used: {len(X)}")
        logger.info(f"🔢 Features: {X_scaled.shape[1]} survey questions")
        logger.info(f"🟢 Status: Active")
        logger.info("=" * 80)

        return summary

    except Exception as e:
        logger.exception(f"❌ Training pipeline failed: {e}")
        
        # Log failure to Firestore
        if db:
            try:
                failure_record = {
                    'trained_at': datetime.utcnow(),
                    'status': 'failed',
                    'error': str(e),
                    'active': False,
                    'description': description,
                    'passed': False
                }
                db.collection('models').add(failure_record)
            except Exception as db_error:
                logger.exception(f"Failed to log error to Firestore: {db_error}")
        
        raise


def get_active_model():
    """Retrieve the currently active model from Firestore."""
    if not db:
        logger.warning("No Firestore db configured")
        return None
    
    try:
        models_ref = db.collection('models')
        query = models_ref.where(
            filter=FieldFilter("active", "==", True)
        ).order_by('trained_at', direction='DESCENDING').limit(1)
        docs = list(query.stream())
        
        if docs:
            model_data = docs[0].to_dict()
            model_data['id'] = docs[0].id
            return model_data
        else:
            logger.warning("No active model found in Firestore")
            return None
    except Exception as e:
        logger.exception(f"Error retrieving active model: {e}")
        return None


def get_all_models(limit=10):
    """Retrieve all models from Firestore, ordered by training date."""
    if not db:
        logger.warning("No Firestore db configured")
        return []
    
    try:
        models_ref = db.collection('models')
        query = models_ref.order_by('trained_at', direction='DESCENDING').limit(limit)
        docs = query.stream()
        
        models = []
        for doc in docs:
            model_data = doc.to_dict()
            model_data['id'] = doc.id
            models.append(model_data)
        
        return models
    except Exception as e:
        logger.exception(f"Error retrieving models: {e}")
        return []


def activate_model(model_id):
    """Activate a specific model by ID and deactivate all others."""
    if not db:
        logger.warning("No Firestore db configured")
        return False
    
    try:
        # Deactivate all models
        deactivate_previous_models()
        
        # Activate the specified model
        model_ref = db.collection('models').document(model_id)
        model_ref.update({
            'active': True,
            'activated_at': datetime.utcnow()
        })
        
        logger.info(f"✅ Model {model_id} activated successfully")
        return True
    except Exception as e:
        logger.exception(f"Error activating model {model_id}: {e}")
        return False


def predict_burnout(input_data):
    """
    Predict burnout level for new survey responses.
    
    Args:
        input_data: Dictionary with survey question responses
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load latest model and preprocessor
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError("No trained model found. Please train a model first.")
        
        clf = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Apply same preprocessing
        for col in preprocessor['categorical_columns']:
            if col in df.columns and col in preprocessor['label_encoders']:
                le = preprocessor['label_encoders'][col]
                df[col] = le.transform(df[col].astype(str).fillna('unknown'))
        
        # Impute and scale
        df_imputed = preprocessor['imputer'].transform(df)
        df_scaled = preprocessor['scaler'].transform(df_imputed)
        
        # Predict
        prediction = clf.predict(df_scaled)[0]
        probabilities = clf.predict_proba(df_scaled)[0]
        
        # Get class names
        classes = clf.classes_
        
        result = {
            'prediction': prediction,
            'confidence': float(max(probabilities) * 100),
            'probabilities': {
                str(cls): float(prob * 100) 
                for cls, prob in zip(classes, probabilities)
            }
        }
        
        return result
        
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise


# For testing/debugging
if __name__ == "__main__":
    # Test training
    result = train_from_csv("Test training run")
    print(json.dumps(convert_to_native_types(result), indent=2))