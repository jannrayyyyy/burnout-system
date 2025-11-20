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
import requests
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
    matthews_corrcoef, balanced_accuracy_score, precision_score, 
    recall_score, f1_score
)
import seaborn as sns
from scipy import stats
from google.cloud.firestore_v1.base_query import FieldFilter
import tempfile
import urllib.parse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

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


def create_requests_session():
    """Create a robust requests session with retry strategy."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set reasonable timeout and headers
    session.headers.update({
        'User-Agent': 'Burnout-Training-Service/1.0',
        'Accept': 'text/csv, application/csv, */*'
    })
    
    return session


def validate_csv_source(csv_source):
    """
    Validate and normalize the CSV source input.
    
    Args:
        csv_source: URL, file path, or Firebase Storage path
        
    Returns:
        dict: Normalized source information
    """
    if csv_source is None:
        return {
            'type': 'default',
            'path': str(DATA_PATH),
            'valid': DATA_PATH.exists()
        }
    
    # Convert to string if Path object
    csv_source = str(csv_source)
    
    # Check if it's a URL
    if csv_source.startswith(('http://', 'https://')):
        return {
            'type': 'url',
            'path': csv_source,
            'valid': True
        }
    
    # Check if it's a Firebase Storage path (gs:// or firebase storage pattern)
    elif csv_source.startswith('gs://') or 'firebasestorage.googleapis.com' in csv_source:
        return {
            'type': 'firebase',
            'path': csv_source,
            'valid': True
        }
    
    # Check if it's a local file path
    else:
        file_path = Path(csv_source)
        return {
            'type': 'local',
            'path': str(file_path),
            'valid': file_path.exists()
        }


def download_from_firebase_storage(firebase_url):
    """
    Download CSV from Firebase Storage URL.
    
    Args:
        firebase_url: Firebase Storage URL
        
    Returns:
        str: CSV content as string
    """
    try:
        logger.info(f"🔥 Downloading from Firebase Storage: {firebase_url}")
        
        session = create_requests_session()
        
        # Handle Firebase Storage URL formatting
        if 'alt=media' not in firebase_url:
            # Ensure the URL has the media parameter
            parsed_url = urllib.parse.urlparse(firebase_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            query_params['alt'] = ['media']
            new_query = urllib.parse.urlencode(query_params, doseq=True)
            firebase_url = urllib.parse.urlunparse((
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                new_query,
                parsed_url.fragment
            ))
        
        response = session.get(firebase_url, timeout=30)
        response.raise_for_status()
        
        # Validate that we got CSV content
        content_type = response.headers.get('content-type', '').lower()
        if 'text/csv' not in content_type and 'application/csv' not in content_type:
            # Check if content looks like CSV
            content_preview = response.text[:100]
            if ',' not in content_preview and '\n' not in content_preview:
                logger.warning(f"⚠️ Response may not be CSV. Content-Type: {content_type}")
        
        logger.info(f"✅ Successfully downloaded CSV from Firebase Storage: {len(response.text)} bytes")
        return response.text
        
    except requests.RequestException as e:
        logger.error(f"❌ Firebase Storage download failed: {e}")
        raise ValueError(f"Failed to download from Firebase Storage: {str(e)}")


def load_csv_from_url_or_path(source):
    """
    Enhanced CSV loading with robust error handling and support for multiple sources.
    
    Args:
        source: URL, file path, or Firebase Storage path
        
    Returns:
        pandas DataFrame
        
    Raises:
        ValueError: If source is invalid or data cannot be loaded
        FileNotFoundError: If local file doesn't exist
    """
    source_info = validate_csv_source(source)
    
    logger.info(f"📥 Loading data from: {source_info['path']} (type: {source_info['type']})")
    
    try:
        if source_info['type'] == 'url':
            # Standard HTTP/HTTPS URL
            session = create_requests_session()
            response = session.get(source_info['path'], timeout=30)
            response.raise_for_status()
            csv_content = response.text
            
        elif source_info['type'] == 'firebase':
            # Firebase Storage URL
            csv_content = download_from_firebase_storage(source_info['path'])
            
        elif source_info['type'] == 'local':
            # Local file path
            if not source_info['valid']:
                raise FileNotFoundError(f"Local file not found: {source_info['path']}")
            with open(source_info['path'], 'r', encoding='utf-8') as f:
                csv_content = f.read()
                
        elif source_info['type'] == 'default':
            # Default data path
            if not source_info['valid']:
                raise FileNotFoundError(f"Default data file not found: {source_info['path']}")
            with open(source_info['path'], 'r', encoding='utf-8') as f:
                csv_content = f.read()
        
        else:
            raise ValueError(f"Unsupported source type: {source_info['type']}")
        
        # Parse CSV content
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            logger.info(f"✅ Successfully loaded CSV: {len(df)} rows × {df.shape[1]} columns")
            return df
            
        except pd.errors.ParserError as e:
            logger.error(f"❌ CSV parsing error: {e}")
            
            # Try alternative encodings for local files
            if source_info['type'] in ['local', 'default']:
                logger.info("🔄 Trying alternative encodings...")
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(source_info['path'], encoding=encoding)
                        logger.info(f"✅ Successfully loaded with {encoding} encoding")
                        return df
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
            
            raise ValueError(f"Failed to parse CSV: {str(e)}")
            
    except requests.RequestException as e:
        logger.error(f"❌ Network error loading CSV: {e}")
        raise ValueError(f"Network error loading CSV: {str(e)}")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error loading CSV: {e}")
        raise ValueError(f"Failed to load CSV: {str(e)}")


def backup_csv_source(csv_content, source_info):
    """
    Backup the CSV source to local storage for reproducibility.
    
    Args:
        csv_content: Raw CSV content as string
        source_info: Source information dictionary
    """
    try:
        backup_dir = Path("data/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_type = source_info['type']
        
        if source_type == 'url':
            # Extract domain for filename
            domain = urllib.parse.urlparse(source_info['path']).netloc
            filename = f"backup_{timestamp}_{domain}.csv"
        elif source_type == 'firebase':
            filename = f"backup_{timestamp}_firebase.csv"
        elif source_type == 'local':
            file_path = Path(source_info['path'])
            filename = f"backup_{timestamp}_{file_path.stem}.csv"
        else:
            filename = f"backup_{timestamp}_unknown.csv"
        
        backup_path = backup_dir / filename
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        logger.info(f"💾 CSV source backed up to: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to backup CSV source: {e}")
        return None


def validate_dataset_structure(df):
    """
    Validate that the dataset has the expected structure for burnout analysis.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        tuple: (is_valid, message)
    """
    if df.empty:
        return False, "Dataset is empty"
    
    if len(df) < 10:
        return False, f"Insufficient rows: {len(df)} (minimum 10 required)"
    
    if df.shape[1] < 5:
        return False, f"Insufficient columns: {df.shape[1]} (minimum 5 required)"
    
    # Check for expected column patterns (survey questions)
    text_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(text_columns) == 0 and len(numeric_columns) == 0:
        return False, "No usable columns found"
    
    logger.info(f"📊 Dataset validation: {len(df)} rows, {len(text_columns)} text columns, {len(numeric_columns)} numeric columns")
    return True, "Dataset structure appears valid"


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
    """Clean and prepare data for training - UNCHANGED FROM ORIGINAL"""
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
    """Map Likert scale responses to numerical values - UNCHANGED FROM ORIGINAL"""
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
    """Derive burnout labels using multi-dimensional analysis - UNCHANGED FROM ORIGINAL"""
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


def create_visualizations(clf, X_test, y_test, y_pred, model_results, 
                         feature_names, class_names, version, best_model_name):
    """
    Create comprehensive visualizations for model evaluation.
    
    Returns:
        dict: Dictionary of BytesIO buffers containing PNG images
    """
    visualizations = {}
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Confusion Matrix
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {best_model_name}\nVersion {version}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        visualizations['confusion_matrix'] = buf
    except Exception as e:
        logger.error(f"Failed to create confusion matrix: {e}")
    
    # 2. Model Comparison
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(model_results.keys())
        accuracies = list(model_results.values())
        colors = [COLOR_PALETTE['success'] if m == best_model_name 
                 else COLOR_PALETTE['secondary'] for m in models]
        
        bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title(f'Model Performance Comparison\nVersion {version}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        visualizations['model_comparison'] = buf
    except Exception as e:
        logger.error(f"Failed to create model comparison: {e}")
    
    # 3. Feature Importance
    if hasattr(clf, 'feature_importances_'):
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            importances = clf.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20
            
            ax.barh(range(len(indices)), importances[indices], 
                   color=COLOR_PALETTE['accent'], edgecolor='black', linewidth=1)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i][:50] for i in indices], fontsize=9)
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_title(f'Top 20 Feature Importances - {best_model_name}\nVersion {version}', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            visualizations['feature_importance'] = buf
        except Exception as e:
            logger.error(f"Failed to create feature importance: {e}")
    
    # 4. Class Distribution
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Actual distribution
        actual_counts = pd.Series(y_test).value_counts()
        ax1.pie(actual_counts.values, labels=actual_counts.index, autopct='%1.1f%%',
               colors=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']],
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Actual Class Distribution\n(Test Set)', fontsize=12, fontweight='bold')
        
        # Predicted distribution
        pred_counts = pd.Series(y_pred).value_counts()
        ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%',
               colors=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']],
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Predicted Class Distribution\n(Test Set)', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'Burnout Level Distribution - Version {version}', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        visualizations['class_distribution'] = buf
    except Exception as e:
        logger.error(f"Failed to create class distribution: {e}")
    
    return visualizations


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive evaluation metrics for classification models.
    Supports multi-class and handles cases with undefined probabilities.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )),
        "recall": float(recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )),
        "f1": float(f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )),
    }

    # AUC calculation (safe)
    if y_proba is not None:
        try:
            metrics["auc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovo")
            )
        except Exception:
            # AUC not supported for this case
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    return metrics


def train_from_csv(description: str = "Burnout prediction model trained on student survey data", 
                   csv_source: str = None):
    """
    Enhanced main training pipeline with robust CSV source handling.
    TRAINING LOGIC UNCHANGED FROM ORIGINAL.
    
    Args:
        description: Model description
        csv_source: URL, file path, or Firebase Storage path to the CSV
        
    Returns:
        dict: Training summary
    """
    source_info = validate_csv_source(csv_source)
    
    try:
        # ========== PHASE 0: DEACTIVATE PREVIOUS MODELS ==========
        deactivate_previous_models()

        # ========== PHASE 1: ENHANCED DATA LOADING ==========
        logger.info("=" * 80)
        logger.info("🚀 STARTING ENHANCED BURNOUT PREDICTION TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"📥 Data Source: {source_info['path']}")
        logger.info(f"📋 Source Type: {source_info['type']}")
        
        # Load data with enhanced error handling
        df_original = load_csv_from_url_or_path(csv_source)
        original_row_count = len(df_original)
        logger.info(f"📂 Loaded dataset: {original_row_count} rows × {df_original.shape[1]} columns")
        
        # Validate dataset structure
        is_valid, validation_msg = validate_dataset_structure(df_original)
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {validation_msg}")
        
        if df_original.empty or original_row_count < 30:
            raise ValueError(f"Insufficient data: {original_row_count} samples (minimum 30 required)")

        # ========== PHASE 2: BACKUP CSV SOURCE ==========
        backup_path = None
        try:
            if source_info['type'] in ['url', 'firebase']:
                session = create_requests_session()
                response = session.get(source_info['path'], timeout=30)
                csv_content = response.text
            else:
                with open(source_info['path'], 'r', encoding='utf-8') as f:
                    csv_content = f.read()
            
            backup_path = backup_csv_source(csv_content, source_info)
        except Exception as backup_error:
            logger.warning(f"⚠️ CSV backup failed: {backup_error}")

        # ========== PHASE 3: DATA PREPROCESSING (UNCHANGED) ==========
        df = clean_and_prepare_data(df_original.copy())
        df = map_likert_responses(df)
        
        # Derive burnout labels
        df, label_col = derive_burnout_labels(df)
        
        # Prepare features and labels
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

        # Scaling
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

        # Train-test split
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

        # ========== PHASE 4: MODEL TRAINING (UNCHANGED) ==========
        logger.info("\n🤖 Training models...")
        
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                class_weight='balanced',
                random_state=42
            ),
            "SVM": SVC(
                kernel='rbf',
                C=0.1,
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

        # ========== PHASE 5: EVALUATION (UNCHANGED) ==========
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        class_names = sorted(set(y_test))

        # Visualizations
        visualizations = create_visualizations(
            clf, X_test, y_test, y_pred, results,
            X.columns.tolist(), class_names, 
            len(list(MODELS_DIR.glob("burnout_v*.pkl"))) + 1,
            best_model_name
        )

        # Feature importance
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

        # ========== PHASE 6: SAVE MODELS (UNCHANGED) ==========
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

        # ========== PHASE 7: FIREBASE UPLOAD (UNCHANGED) ==========
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

        # ========== PHASE 8: FIRESTORE RECORD ==========
        record = {
            'version': version,
            'trained_at': datetime.utcnow(),
            'description': description,
            'data_source': source_info['path'],
            'data_source_type': source_info['type'],
            'backup_path': backup_path,
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

        # ========== PHASE 9: SUMMARY ==========
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
            'data_source': source_info['path'],
            'data_source_type': source_info['type'],
            'backup_path': backup_path,
            'original_row_count': original_row_count,
            'records_used': len(X),
            'n_features': X_scaled.shape[1],
            'active': True
        }

        logger.info("\n" + "=" * 80)
        logger.info("✅ ENHANCED TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📦 Model Version: {version}")
        logger.info(f"🏆 Best Model: {best_model_name}")
        logger.info(f"🎯 Test Accuracy: {best_accuracy:.2f}%")
        logger.info(f"📥 Data Source: {source_info['path']}")
        logger.info(f"📋 Source Type: {source_info['type']}")
        logger.info(f"💾 Backup: {backup_path or 'Not available'}")
        logger.info(f"📈 Original Records: {original_row_count}")
        logger.info(f"✅ Records Used: {len(X)}")
        logger.info(f"🔢 Features: {X_scaled.shape[1]} survey questions")
        logger.info(f"🟢 Status: Active")
        logger.info("=" * 80)

        return summary

    except Exception as e:
        logger.exception(f"❌ Enhanced training pipeline failed: {e}")
        
        # Enhanced failure logging
        if db:
            try:
                failure_record = {
                    'trained_at': datetime.utcnow(),
                    'status': 'failed',
                    'error': str(e),
                    'data_source': source_info['path'],
                    'data_source_type': source_info['type'],
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
    """Retrieve all models from Firestore with pagination."""
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
    """Activate a specific model and deactivate all others."""
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
    Make burnout prediction using the active model.
    
    Args:
        input_data: Dictionary of survey responses
        
    Returns:
        dict: Prediction results with confidence scores
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


def get_model_statistics():
    """
    Get comprehensive statistics about all trained models.
    
    Returns:
        dict: Model statistics and analytics
    """
    if not db:
        logger.warning("No Firestore db configured")
        return {}
    
    try:
        models = get_all_models(limit=100)
        
        if not models:
            return {
                'total_models': 0,
                'active_models': 0,
                'average_accuracy': 0,
                'best_model': None
            }
        
        active_models = [m for m in models if m.get('active', False)]
        successful_models = [m for m in models if m.get('status') == 'completed']
        
        accuracies = [m.get('accuracy', 0) for m in successful_models]
        best_model = max(successful_models, key=lambda m: m.get('accuracy', 0)) if successful_models else None
        
        # Model type distribution
        model_types = {}
        for m in successful_models:
            model_name = m.get('best_model', 'Unknown')
            model_types[model_name] = model_types.get(model_name, 0) + 1
        
        stats = {
            'total_models': len(models),
            'active_models': len(active_models),
            'successful_models': len(successful_models),
            'failed_models': len([m for m in models if m.get('status') == 'failed']),
            'average_accuracy': float(np.mean(accuracies)) if accuracies else 0,
            'max_accuracy': float(max(accuracies)) if accuracies else 0,
            'min_accuracy': float(min(accuracies)) if accuracies else 0,
            'std_accuracy': float(np.std(accuracies)) if accuracies else 0,
            'best_model': {
                'id': best_model.get('id'),
                'version': best_model.get('version'),
                'accuracy': best_model.get('accuracy'),
                'model_name': best_model.get('best_model'),
                'trained_at': best_model.get('trained_at')
            } if best_model else None,
            'model_type_distribution': model_types,
            'recent_models': [
                {
                    'id': m.get('id'),
                    'version': m.get('version'),
                    'accuracy': m.get('accuracy'),
                    'model_name': m.get('best_model'),
                    'trained_at': m.get('trained_at'),
                    'active': m.get('active', False)
                }
                for m in successful_models[:5]
            ]
        }
        
        return stats
        
    except Exception as e:
        logger.exception(f"Error getting model statistics: {e}")
        return {}


def delete_model(model_id):
    """
    Delete a model from Firestore and optionally from Storage.
    
    Args:
        model_id: Firestore document ID of the model to delete
        
    Returns:
        bool: Success status
    """
    if not db:
        logger.warning("No Firestore db configured")
        return False
    
    try:
        # Get model data first
        model_ref = db.collection('models').document(model_id)
        model_doc = model_ref.get()
        
        if not model_doc.exists:
            logger.warning(f"Model {model_id} not found")
            return False
        
        model_data = model_doc.to_dict()
        
        # Don't allow deletion of active model
        if model_data.get('active', False):
            logger.warning(f"Cannot delete active model {model_id}")
            return False
        
        # Delete from Firestore
        model_ref.delete()
        logger.info(f"✅ Model {model_id} deleted from Firestore")
        
        # Optionally delete from Storage
        if bucket:
            try:
                version = model_data.get('version')
                if version:
                    # Delete model file
                    model_blob = bucket.blob(f"models/burnout_v{version}.pkl")
                    if model_blob.exists():
                        model_blob.delete()
                    
                    # Delete preprocessor
                    preprocessor_blob = bucket.blob(f"models/preprocessor_v{version}.pkl")
                    if preprocessor_blob.exists():
                        preprocessor_blob.delete()
                    
                    # Delete visualizations
                    viz_blobs = bucket.list_blobs(prefix=f"visualizations/burnout_v{version}/")
                    for blob in viz_blobs:
                        blob.delete()
                    
                    logger.info(f"✅ Model {model_id} files deleted from Storage")
            except Exception as storage_error:
                logger.warning(f"⚠️ Error deleting storage files: {storage_error}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error deleting model {model_id}: {e}")
        return False


# Enhanced testing function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ENHANCED TRAINING SERVICE - TEST SUITE")
    print("="*80)
    
    # Test with various CSV sources
    test_sources = [
        {
            'name': 'Firebase Storage',
            'source': "https://firebasestorage.googleapis.com/v0/b/burnout-system.firebasestorage.app/o/csv%2Fburnout_data.csv?alt=media&token=ffd5823b-5880-43bf-8a2d-d84c7db58522"
        },
        {
            'name': 'Local File',
            'source': "data/burnout_data.csv"
        },
        {
            'name': 'Default (None)',
            'source': None
        }
    ]
    
    for test_case in test_sources:
        try:
            print(f"\n{'='*60}")
            print(f"Testing: {test_case['name']}")
            print(f"Source: {test_case['source']}")
            print(f"{'='*60}")
            
            result = train_from_csv(
                description=f"Test training run - {test_case['name']}",
                csv_source=test_case['source']
            )
            
            print("\n✅ TRAINING SUCCESSFUL")
            print(f"Version: {result.get('version')}")
            print(f"Best Model: {result.get('best_model')}")
            print(f"Accuracy: {result.get('accuracy'):.2f}%")
            print(f"Data Source Type: {result.get('data_source_type')}")
            print(f"Records Used: {result.get('records_used')}/{result.get('original_row_count')}")
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Test model statistics
    print(f"\n{'='*60}")
    print("Testing: Model Statistics")
    print(f"{'='*60}")
    
    try:
        stats = get_model_statistics()
        print(json.dumps(convert_to_native_types(stats), indent=2, default=str))
    except Exception as e:
        print(f"❌ Statistics failed: {e}")
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)