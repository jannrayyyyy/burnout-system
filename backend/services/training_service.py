# backend/services/training_service.py
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import io
import json
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, learning_curve
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)
import seaborn as sns
from scipy import stats

from .firebase_service import db, bucket

logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path("data/burnout_data.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "burnout_latest.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_latest.pkl"

# color scheme for visualizations sa charts
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
    # --- ADD THIS NEW BLOCK ---
    elif isinstance(obj, pd.Series):
        # Convert series to list, then recursively clean the list's contents
        return convert_to_native_types(obj.tolist())
    # --- END NEW BLOCK ---
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


def clean_and_normalize_data(df):
    """Advanced data cleaning with outlier detection and normalization."""
    logger.info("🧹 Starting data cleaning and normalization...")
    
    # Normalize column names
    df.columns = [
        c.strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        for c in df.columns
    ]
    
    # Drop unnecessary columns
    drop_cols = [
        "timestamp", "name", "institution", "trained_at", 
        "model_version", "status", "id", "user_id", "unnamed"
    ]
    df.drop(columns=[c for c in df.columns if any(d in c for d in drop_cols)], 
            inplace=True, errors="ignore")
    
    # Enhanced empty value cleaning
    empty_values = ["", " ", "nan", "NaN", "NA", "N/A", "null", "None", "#N/A", "?", "--"]
    df.replace(empty_values, np.nan, inplace=True)
    
    # Remove completely empty rows
    df.dropna(how='all', inplace=True)
    
    logger.info(f"✓ Data cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def advanced_likert_mapping(df):
    """Enhanced Likert scale mapping with fuzzy matching."""
    logger.info("📊 Applying advanced Likert scale mapping...")
    
    # Comprehensive Likert mappings
    likert_map = {
        # Standard 5-point scale
        "strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5,
        # Common variations
        "strongly_disagree": 1, "strongly_agree": 5,
        # Typos and variations
        "argee": 4, "agre": 4, "neural": 3, "nuetral": 3,
        "disargee": 2, "disagre": 2, "strongly argee": 5, "strongly disagre": 1,
        # Frequency scale
        "never": 1, "rarely": 2, "sometimes": 3, "often": 4, "always": 5,
        # Intensity scale
        "very low": 1, "low": 2, "medium": 3, "moderate": 3, "high": 4, "very high": 5,
        # Binary
        "no": 1, "yes": 5,
    }
    
    likert_applied = 0
    for col in df.select_dtypes(include=["object"]).columns:
        original_values = df[col].copy()
        df[col] = df[col].apply(
            lambda v: likert_map.get(str(v).strip().lower(), v) if pd.notna(v) else v
        )
        if not df[col].equals(original_values):
            likert_applied += 1
    
    logger.info(f"✓ Likert mapping applied to {likert_applied} columns")
    return df


def derive_burnout_labels(df):
    """Intelligent burnout label derivation with multi-dimensional analysis."""
    logger.info("🔍 Deriving burnout labels from multi-dimensional analysis...")
    
    # Define burnout dimension keywords
    dimensions = {
        'exhaustion': ['sleep', 'fatigue', 'exhausted', 'tired', 'energy', 'rest'],
        'cynicism': ['motivation', 'interest', 'excited', 'giving_up', 'accomplishment'],
        'inefficacy': ['performance', 'competent', 'achievement', 'confidence', 'underperforming'],
        'stress': ['stress', 'pressure', 'overwhelm', 'anxiety', 'worry', 'dread'],
        'workload': ['workload', 'academic', 'deadline', 'task', 'responsibility']
    }
    
    dimension_scores = {}
    
    for dimension, keywords in dimensions.items():
        related_cols = [
            col for col in df.columns
            if any(kw in col.lower() for kw in keywords)
        ]
        if related_cols:
            dimension_scores[dimension] = df[related_cols].apply(
                pd.to_numeric, errors='coerce'
            ).mean(axis=1)
    
    if not dimension_scores:
        # Fallback: use all numeric columns
        logger.warning("⚠️ No dimension-specific columns found. Using all numeric features.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        burnout_index = df[numeric_cols].mean(axis=1)
    else:
        # Weighted composite score
        weights = {'exhaustion': 0.25, 'cynicism': 0.20, 'inefficacy': 0.20, 
                   'stress': 0.20, 'workload': 0.15}
        burnout_index = sum(
            dimension_scores[dim] * weights.get(dim, 0.2) 
            for dim in dimension_scores
        ) / sum(weights.get(dim, 0.2) for dim in dimension_scores)
    
    # Advanced thresholding with statistical analysis
    q25, q50, q75 = burnout_index.quantile([0.25, 0.5, 0.75])
    mean, std = burnout_index.mean(), burnout_index.std()
    
    # Use both percentile and standard deviation for robust classification
    low_threshold = min(q25, mean - 0.5 * std)
    high_threshold = max(q75, mean + 0.5 * std)
    
    conditions = [
        (burnout_index <= low_threshold),
        (burnout_index > low_threshold) & (burnout_index <= high_threshold),
        (burnout_index > high_threshold)
    ]
    choices = ["Low", "Moderate", "High"]
    
    df["burnout_level"] = np.select(conditions, choices, default="Moderate")
    
    # Log distribution
    distribution = df["burnout_level"].value_counts().to_dict()
    logger.info(f"✓ Burnout distribution: {distribution}")
    logger.info(f"  Thresholds: Low ≤ {low_threshold:.2f}, High > {high_threshold:.2f}")
    
    return df, "burnout_level", dimension_scores


def detect_and_remove_outliers(X, threshold=3.0):
    """Statistical outlier detection using Z-score method."""
    logger.info(f"🔍 Detecting outliers (threshold: {threshold} std)...")
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(X[numeric_cols], nan_policy='omit'))
    
    outlier_mask = (z_scores < threshold).all(axis=1)
    n_outliers = (~outlier_mask).sum()
    
    logger.info(f"✓ Found {n_outliers} outlier samples ({n_outliers/len(X)*100:.1f}%)")
    
    return X[outlier_mask], outlier_mask


def create_advanced_visualizations(clf, X_test, y_test, y_pred, results, feature_names, class_names, version, best_model_name):
    """Generate comprehensive visualizations with professional styling."""
    logger.info("📊 Creating advanced visualizations...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    visualizations = {}
    
    # 1. Enhanced Confusion Matrix with percentages
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annot = np.array([[f'{count}\n({pct:.1f}%)' 
                       for count, pct in zip(row_counts, row_pcts)]
                      for row_counts, row_pcts in zip(cm, cm_percent)])
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='RdYlGn', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')
    
    ax1.set_title('Confusion Matrix with Percentages\n(True vs Predicted Labels)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    buf1.seek(0)
    visualizations['confusion_matrix'] = buf1
    
    # 2. Model Comparison with detailed metrics
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = list(results.values())
    colors_comp = [COLOR_PALETTE['success'] if acc == max(accuracies) 
                   else COLOR_PALETTE['secondary'] for acc in accuracies]
    
    bars = ax2a.barh(models, accuracies, color=colors_comp, edgecolor='black', linewidth=1.5)
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax2a.text(acc + 1, i, f'{acc:.2f}%', va='center', fontweight='bold', fontsize=11)
    
    ax2a.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax2a.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2a.set_xlim(0, 105)
    ax2a.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Relative performance
    acc_diff = [acc - min(accuracies) for acc in accuracies]
    ax2b.bar(models, acc_diff, color=COLOR_PALETTE['accent'], 
             edgecolor='black', linewidth=1.5, alpha=0.7)
    ax2b.set_ylabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
    ax2b.set_title('Relative Performance', fontsize=14, fontweight='bold')
    ax2b.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, diff in enumerate(acc_diff):
        ax2b.text(i, diff + 0.5, f'+{diff:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    buf2.seek(0)
    visualizations['model_comparison'] = buf2
    
    # 3. Feature Importance (if available)
    if hasattr(clf, 'feature_importances_'):
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        
        feat_imp = sorted(zip(feature_names, clf.feature_importances_), 
                         key=lambda x: x[1], reverse=True)[:15]
        features, importances = zip(*feat_imp)
        
        colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = ax3.barh(range(len(features)), importances, color=colors_feat,
                       edgecolor='black', linewidth=1.2)
        
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
        ax3.invert_yaxis()
        ax3.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
        title = f'Top 15 Feature Importances\n(Model: {best_model_name})'
        ax3.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax3.text(imp + 0.002, i, f'{imp:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        buf3.seek(0)
        visualizations['feature_importance'] = buf3
    
    # 4. Class Distribution
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    class_counts = pd.Series(y_test).value_counts()
    colors_pie = [COLOR_PALETTE['success'], COLOR_PALETTE['accent'], COLOR_PALETTE['danger']]
    wedges, texts, autotexts = ax4a.pie(
        class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
        colors=colors_pie, explode=[0.05]*len(class_counts),
        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    ax4a.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart with predictions
    x_pos = np.arange(len(class_names))
    true_counts = [list(y_test).count(c) for c in class_names]
    pred_counts = [list(y_pred).count(c) for c in class_names]
    
    width = 0.35
    ax4b.bar(x_pos - width/2, true_counts, width, label='True', 
            color=COLOR_PALETTE['secondary'], edgecolor='black')
    ax4b.bar(x_pos + width/2, pred_counts, width, label='Predicted',
            color=COLOR_PALETTE['success'], edgecolor='black', alpha=0.7)
    
    ax4b.set_xlabel('Burnout Level', fontsize=13, fontweight='bold')
    ax4b.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax4b.set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
    ax4b.set_xticks(x_pos)
    ax4b.set_xticklabels(class_names)
    ax4b.legend(fontsize=11)
    ax4b.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    buf4 = io.BytesIO()
    plt.savefig(buf4, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    buf4.seek(0)
    visualizations['class_distribution'] = buf4
    
    logger.info(f"✓ Generated {len(visualizations)} visualizations")
    return visualizations


def calculate_comprehensive_metrics(y_test, y_pred, y_proba=None):
    """Calculate extensive evaluation metrics."""
    logger.info("📈 Calculating comprehensive metrics...")
    
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
    metrics['macro_precision'] = float(precision.mean()) * 100
    metrics['macro_recall'] = float(recall.mean()) * 100
    metrics['macro_f1'] = float(f1.mean()) * 100
    
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_precision'] = float(precision_w) * 100
    metrics['weighted_recall'] = float(recall_w) * 100
    metrics['weighted_f1'] = float(f1_w) * 100
    
    # Additional metrics
    metrics['cohen_kappa'] = float(cohen_kappa_score(y_test, y_pred))
    metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_test, y_pred))
    
    # ROC AUC (if probabilities available)
    if y_proba is not None and len(classes) > 2:
        try:
            metrics['roc_auc_ovr'] = float(roc_auc_score(
                y_test, y_proba, multi_class='ovr', average='weighted'
            ))
        except:
            pass
    
    logger.info(f"✓ Accuracy: {metrics['accuracy']:.2f}% | F1: {metrics['weighted_f1']:.2f}%")
    return metrics


def train_from_csv(description: str = "Enhanced burnout prediction with advanced analytics"):
    """
    Production-grade training pipeline with:
    - Advanced preprocessing
    - Multi-model comparison
    - Comprehensive evaluation
    - Professional visualizations
    - Detailed Firebase logging
    """
    
    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

        # ========== PHASE 1: DATA LOADING ==========
        logger.info("=" * 80)
        logger.info("🚀 STARTING ENHANCED TRAINING PIPELINE")
        logger.info("=" * 80)
        
        df = pd.read_csv(DATA_PATH)
        logger.info(f"📊 Initial dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if df.empty or len(df) < 30:
            raise ValueError(f"Insufficient data: {len(df)} samples (minimum 30 required)")

        # ========== PHASE 2: PREPROCESSING ==========
        df = clean_and_normalize_data(df)
        df = advanced_likert_mapping(df)
        
        # ========== PHASE 3: LABEL DETECTION/DERIVATION ==========
        label_candidates = ["prediction", "label", "burnout_level", "burnout", "target"]
        label_col = next((c for c in label_candidates if c in df.columns), None)
        
        if label_col is None:
            df, label_col, dimension_scores = derive_burnout_labels(df)
            derived_label = True
        else:
            logger.info(f"✅ Found existing label column: '{label_col}'")
            derived_label = False
            dimension_scores = {}

        # ========== PHASE 4: FEATURE ENGINEERING ==========
        X = df.drop(columns=[label_col])
        y = df[label_col].astype(str).str.strip()
        
        # Remove invalid labels
        valid_mask = y.notna() & (y != '') & (y != 'nan')
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) < 30:
            raise ValueError(f"Insufficient valid samples: {len(X)}")
        
        logger.info(f"✓ Features: {X.shape[1]} | Samples: {len(X)}")
        logger.info(f"✓ Label distribution: {dict(y.value_counts())}")

        # Encode categorical variables
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        label_encoders = {}
        
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))
            label_encoders[col] = le

        # Outlier removal (optional, can be disabled)
        original_size = len(X)
        X, outlier_mask = detect_and_remove_outliers(X, threshold=3.5)
        y = y[outlier_mask]
        logger.info(f"✓ Removed {original_size - len(X)} outliers")

        # Imputation
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Robust scaling (more resistant to outliers than StandardScaler)
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

        # Save preprocessor
        preprocessor = {
            'label_encoders': label_encoders,
            'imputer': imputer,
            'scaler': scaler,
            'feature_names': X.columns.tolist(),
            'categorical_columns': cat_cols,
            'derived_label': derived_label
        }

        # ========== PHASE 5: TRAIN-TEST SPLIT ==========
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        
        logger.info(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

        # ========== PHASE 6: MODEL TRAINING & COMPARISON ==========
        logger.info("\n🎯 Training multiple models...")
        
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=3,
                class_weight='balanced',
                random_state=42
            ),
            "SVM (RBF)": SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }

        results = {}
        trained_models = {}
        cv_scores = {}
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"\n  Training {name}...")
            
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
            cv_scores[name] = {'mean': cv_mean, 'std': cv_std, 'scores': cv_acc.tolist()}
            
            logger.info(f"    ✓ Test Accuracy: {acc:.2f}%")
            logger.info(f"    ✓ CV Accuracy: {cv_mean:.2f}% ± {cv_std:.2f}%")
            
            # OOB score for Random Forest
            if hasattr(model, 'oob_score_'):
                logger.info(f"    ✓ OOB Score: {model.oob_score_ * 100:.2f}%")

        # Select best model
        best_model_name = max(results, key=results.get)
        clf = trained_models[best_model_name]
        best_accuracy = results[best_model_name]
        
        logger.info(f"\n🏆 Best Model: {best_model_name} ({best_accuracy:.2f}%)")

        # ========== PHASE 7: DETAILED EVALUATION ==========
        logger.info("\n📊 Generating comprehensive evaluation...")
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
        
        # Comprehensive metrics
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix details
        cm = confusion_matrix(y_test, y_pred)
        class_names = sorted(set(y_test))

        # ========== PHASE 8: VISUALIZATIONS ==========
        visualizations = create_advanced_visualizations(
            clf, X_test, y_test, y_pred, results,
            X.columns.tolist(), class_names, 
            len(list(MODELS_DIR.glob("burnout_v*.pkl"))) + 1,
            best_model_name  # <-- ADD THIS
        )

        # ========== PHASE 9: FEATURE IMPORTANCE ==========
        important_features = []
        if hasattr(clf, 'feature_importances_'):
            feat_imp = sorted(
                zip(X.columns, clf.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:20]
            important_features = [
                {'feature': str(name), 'importance': float(imp)}
                for name, imp in feat_imp
            ]

        # ========== PHASE 10: MODEL PERSISTENCE ==========
        version = len(list(MODELS_DIR.glob("burnout_v*.pkl"))) + 1
        version_file = MODELS_DIR / f"burnout_v{version}.pkl"
        
        joblib.dump(clf, version_file)
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        
        logger.info(f"💾 Models saved: {version_file}")

        # Save enhanced dataset
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
                try:
                    model_blob.make_public()
                    urls['model'] = model_blob.public_url
                except Exception:
                    urls['model'] = None
                    logger.warning("Could not make model blob public; continuing.")
                
                # Preprocessor file
                preprocessor_blob = bucket.blob(f"models/preprocessor_v{version}.pkl")
                preprocessor_blob.upload_from_filename(str(PREPROCESSOR_PATH))
                try:
                    preprocessor_blob.make_public()
                    urls['preprocessor'] = preprocessor_blob.public_url
                except Exception:
                    urls['preprocessor'] = None

                # Dataset
                dataset_blob = bucket.blob(f"datasets/burnout_labeled_v{version}.csv")
                dataset_blob.upload_from_filename(str(dataset_path))
                try:
                    dataset_blob.make_public()
                    urls['dataset'] = dataset_blob.public_url
                except Exception:
                    urls['dataset'] = None

                # Visualizations (in-memory)
                urls['visualizations'] = {}
                for name, buf in visualizations.items():
                    blob_path = f"visualizations/burnout_v{version}/{name}.png"
                    viz_blob = bucket.blob(blob_path)
                    buf.seek(0)
                    viz_blob.upload_from_string(buf.getvalue(), content_type='image/png')
                    try:
                        viz_blob.make_public()
                        urls['visualizations'][name] = viz_blob.public_url
                    except Exception:
                        urls['visualizations'][name] = None

                # Metrics and metadata (JSON)
                metadata = {
                    'version': version,
                    'trained_at': datetime.utcnow().isoformat() + 'Z',
                    'description': description,
                    'best_model': best_model_name,
                    'accuracy': best_accuracy,
                    'cv_scores': cv_scores,
                    'metrics': metrics,
                    'important_features': important_features,
                    'derived_label': derived_label,
                    'dimension_scores_available': bool(dimension_scores),
                    'sample_count': int(len(X_scaled)),
                }
                metrics_blob = bucket.blob(f"models/burnout_v{version}_metadata.json")
                metrics_blob.upload_from_string(json.dumps(convert_to_native_types(metadata)), content_type='application/json')
                try:
                    metrics_blob.make_public()
                    urls['metadata'] = metrics_blob.public_url
                except Exception:
                    urls['metadata'] = None

                logger.info("✓ Uploads to Firebase Storage complete.")
            except Exception as e:
                logger.exception("Failed uploading artifacts to Firebase Storage: %s", e)
        else:
            logger.warning("No Firebase bucket configured; skipping storage upload.")

        # ========== PHASE 12: FIRESTORE RECORD ==========
        record = {
            'version': version,
            'trained_at': datetime.utcnow(),
            'description': description,
            'best_model': best_model_name,
            'accuracy': float(best_accuracy),
            'metrics': metrics,
            'cv_scores': cv_scores,
            'important_features': important_features,
            'visualization_urls': urls.get('visualizations', {}),
            'model_url': urls.get('model'),
            'preprocessor_url': urls.get('preprocessor'),
            'dataset_url': urls.get('dataset'),
            'metadata_url': urls.get('metadata'),
            'derived_label': derived_label,
            'dimension_scores': convert_to_native_types(dimension_scores),
        }

        try:
            if db:
                logger.info("\n Saving training metadata to Firestore...")
                db.collection('models').add(convert_to_native_types(record))
                logger.info("✓ Firestore record saved.")
            else:
                logger.warning("No Firestore db configured; skipping Firestore save.")
        except Exception as e:
            logger.exception("Failed to save record to Firestore: %s", e)

        # ========== PHASE 13: FINALIZE ==========
        summary = {
            'version': version,
            'best_model': best_model_name,
            'accuracy': best_accuracy,
            'metrics': metrics,
            'important_features': important_features,
            'urls': urls,
            'records_used': int(len(X_scaled)),
            'passed': True,
        }

        logger.info("\n✅ TRAINING PIPELINE COMPLETE")
        logger.info(f"Summary: {json.dumps(convert_to_native_types(summary), indent=2)}")

        return summary

    except Exception as e:
        logger.exception("Training pipeline failed: %s", e)
        raise
