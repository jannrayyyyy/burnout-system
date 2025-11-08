# backend/services/prediction_service.py
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from backend.services.firebase_service import db, bucket

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Model locations
MODELS_DIR = Path("models")
MODEL_LATEST = MODELS_DIR / "burnout_latest.pkl"
PREPROCESSOR_LATEST = MODELS_DIR / "preprocessor_latest.pkl"
TRAINING_HISTORY = MODELS_DIR / "training_history.json"

# Enhanced Likert mapping (EXACT MATCH to training_service)
LIKERT_MAP = {
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
    # Numeric
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
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


def load_model_and_preprocessor() -> Tuple[Optional[BaseEstimator], Optional[Dict], Optional[Dict]]:
    """
    Load the trained model and preprocessor from Firebase Storage.
    Fetches the active model from Firestore and downloads from Storage.
    
    Returns:
        Tuple of (model, preprocessor, metadata)
    """
    try:
        # First, try to get active model from Firebase
        if db:
            try:
                logger.info("Fetching active model from Firebase...")
                models_ref = db.collection('models').where('active', '==', True).order_by('trained_at', direction='DESCENDING').limit(1)
                models = list(models_ref.stream())
                
                if models:
                    model_doc = models[0]
                    model_data = model_doc.to_dict()
                    
                    logger.info(f"Found active model: version {model_data.get('version')}")
                    
                    # Download model and preprocessor from Firebase Storage
                    model_url = model_data.get('model_url')
                    preprocessor_url = model_data.get('preprocessor_url')
                    
                    if model_url and preprocessor_url and bucket:
                        # Extract blob paths from URLs
                        model_blob_path = model_url.split(f"{bucket.name}/")[-1]
                        preprocessor_blob_path = preprocessor_url.split(f"{bucket.name}/")[-1]
                        
                        # Download to temporary files
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as model_tmp:
                            model_blob = bucket.blob(model_blob_path)
                            model_blob.download_to_filename(model_tmp.name)
                            model = joblib.load(model_tmp.name)
                            logger.info(f"Model downloaded from Firebase: {model_blob_path}")
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as prep_tmp:
                            prep_blob = bucket.blob(preprocessor_blob_path)
                            prep_blob.download_to_filename(prep_tmp.name)
                            preprocessor = joblib.load(prep_tmp.name)
                            logger.info(f"Preprocessor downloaded from Firebase: {preprocessor_blob_path}")
                        
                        # Build comprehensive metadata from Firebase
                        metadata = {
                            'version': model_data.get('version'),
                            'model_type': type(model).__name__,
                            'best_model': model_data.get('best_model'),
                            'source': 'firebase',
                            'firebase_id': model_doc.id,
                            'trained_at': model_data.get('trained_at'),
                            'description': model_data.get('description'),
                            'records_used': model_data.get('records_used'),
                            'n_features': model_data.get('n_features'),
                            'n_train_samples': model_data.get('n_train_samples'),
                            'n_test_samples': model_data.get('n_test_samples'),
                            'original_row_count': model_data.get('original_row_count'),
                            'class_distribution': model_data.get('class_distribution'),
                            'cv_scores': model_data.get('cv_scores'),
                            'model_comparison': model_data.get('model_comparison'),
                            'training_metrics': {
                                'accuracy': model_data.get('metrics', {}).get('accuracy'),
                                'balanced_accuracy': model_data.get('metrics', {}).get('balanced_accuracy'),
                                'cohen_kappa': model_data.get('metrics', {}).get('cohen_kappa'),
                                'matthews_corrcoef': model_data.get('metrics', {}).get('matthews_corrcoef'),
                                'weighted_precision': model_data.get('metrics', {}).get('weighted_precision'),
                                'weighted_recall': model_data.get('metrics', {}).get('weighted_recall'),
                                'weighted_f1': model_data.get('metrics', {}).get('weighted_f1'),
                                'per_class': model_data.get('metrics', {}).get('per_class'),
                                'confusion_matrix': model_data.get('metrics', {}).get('confusion_matrix'),
                            },
                            'visualization_urls': model_data.get('visualization_urls'),
                            'model_url': model_url,
                            'preprocessor_url': preprocessor_url,
                        }
                        
                        logger.info(f"Model loaded successfully from Firebase: {metadata['best_model']} v{metadata['version']}")
                        return model, preprocessor, metadata
                        
            except Exception as e:
                logger.warning(f"Could not load model from Firebase: {e}. Falling back to local files.")
        
        # Fallback to local files
        if not MODEL_LATEST.exists() or not PREPROCESSOR_LATEST.exists():
            logger.error("Model or preprocessor files not found locally")
            return None, None, None
        
        logger.info("Loading model from local files...")
        model = joblib.load(MODEL_LATEST)
        preprocessor = joblib.load(PREPROCESSOR_LATEST)
        
        # Load training history if available
        metadata = {
            'version': 'latest',
            'model_type': type(model).__name__,
            'source': 'local_file',
        }
        
        if TRAINING_HISTORY.exists():
            try:
                with open(TRAINING_HISTORY, 'r') as f:
                    history = json.load(f)
                    metadata['training_metrics'] = history.get('metrics', {})
                    metadata['trained_at'] = history.get('timestamp')
                    metadata['version'] = history.get('version')
                    metadata['records_used'] = history.get('records_used')
                    metadata['n_features'] = history.get('n_features')
                    metadata['n_train_samples'] = history.get('n_train_samples')
                    metadata['n_test_samples'] = history.get('n_test_samples')
                    metadata['class_distribution'] = history.get('class_distribution')
                    metadata['cv_scores'] = history.get('cv_scores')
                    metadata['model_comparison'] = history.get('model_comparison')
                    metadata['best_model'] = history.get('best_model')
            except Exception as e:
                logger.warning(f"Could not load training history: {e}")
        
        logger.info(f"Model loaded successfully from local: {type(model).__name__}")
        
        return model, preprocessor, metadata
        
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        return None, None, None


def normalize_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize input payload - convert Likert strings to numbers.
    
    Args:
        payload: Raw survey responses
        
    Returns:
        Normalized dict with numeric values
    """
    normalized = {}
    empty_values = ["", " ", "nan", "NaN", "NA", "N/A", "null", "None", "#N/A", "?", "--"]
    
    for key, value in payload.items():
        # Normalize key
        clean_key = key.strip().lower()
        clean_key = clean_key.replace(" ", "_").replace("-", "_")
        clean_key = clean_key.replace("(", "").replace(")", "")
        clean_key = clean_key.replace("[", "").replace("]", "")
        
        # Handle empty values
        if value in empty_values:
            normalized[clean_key] = np.nan
            continue
        
        # Handle string values
        if isinstance(value, str):
            vs = value.strip()
            
            # Try Likert mapping
            mapped = LIKERT_MAP.get(vs.lower())
            if mapped is not None:
                normalized[clean_key] = mapped
                continue
            
            # Try numeric conversion
            try:
                normalized[clean_key] = float(vs.replace(",", ""))
                continue
            except (ValueError, AttributeError):
                pass
            
            normalized[clean_key] = vs
        elif value is None or (isinstance(value, float) and np.isnan(value)):
            normalized[clean_key] = np.nan
        else:
            normalized[clean_key] = value
    
    return normalized


def get_feature_importance(model: BaseEstimator, feature_names: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
    """Extract feature importance from the model."""
    try:
        # Try to get feature importance from the model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficient values
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            return []
        
        # Create list of (feature, importance) tuples
        feature_importance = [
            {
                "feature": feature_names[i],
                "importance": float(importances[i]),
                "importance_percentage": float(importances[i] / importances.sum() * 100)
            }
            for i in range(len(feature_names))
        ]
        
        # Sort by importance and return top N
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        return feature_importance[:top_n]
        
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return []


def analyze_input_features(input_data: Dict[str, Any], preprocessor: Dict) -> Dict[str, Any]:
    """Analyze the input features to understand what's driving the prediction."""
    analysis = {
        "high_risk_indicators": [],
        "moderate_risk_indicators": [],
        "protective_factors": [],
        "missing_features": [],
        "feature_scores": {}
    }
    
    feature_names = preprocessor['feature_names']
    
    # Define thresholds for burnout indicators (customize based on your features)
    high_risk_features = {
        'emotional_exhaustion': 4,
        'cynicism': 4,
        'workload': 4,
        'stress_level': 4,
        'work_life_balance': 2,  # Lower is worse
        'job_satisfaction': 2,
        'sleep': 2,
        'physical_health': 2,
        'mental_health': 2,
        'anxiety': 4,
        'depression': 4,
    }
    
    moderate_risk_features = {
        'emotional_exhaustion': 3,
        'cynicism': 3,
        'workload': 3,
        'stress_level': 3,
        'sleep': 3,
        'physical_health': 3,
        'anxiety': 3,
    }
    
    protective_features = {
        'social_support': 4,
        'autonomy': 4,
        'job_satisfaction': 4,
        'work_life_balance': 4,
        'sleep': 4,
        'physical_health': 4,
        'mental_health': 4,
        'exercise': 4,
        'relaxation': 4,
    }
    
    # Categorize features by domain
    feature_categories = {
        'sleep': ['sleep', 'sleep_quality', 'sleep_pattern', 'sleep_hours', 'rest'],
        'physical_health': ['physical_health', 'physical', 'exercise', 'fitness', 'health_physical'],
        'mental_health': ['mental_health', 'mental', 'anxiety', 'depression', 'mood', 'emotional_exhaustion'],
        'workload': ['workload', 'work_hours', 'overtime', 'deadlines', 'work_pressure'],
        'social_support': ['social_support', 'support', 'friends', 'family', 'relationships'],
        'work_life_balance': ['work_life_balance', 'balance', 'life_balance'],
        'job_satisfaction': ['job_satisfaction', 'satisfaction', 'fulfillment', 'engagement'],
        'stress': ['stress', 'stress_level', 'tension', 'pressure'],
        'autonomy': ['autonomy', 'control', 'independence', 'flexibility'],
    }
    
    for feature in feature_names:
        value = input_data.get(feature)
        
        if value is None or (isinstance(value, float) and np.isnan(value)):
            analysis['missing_features'].append(feature)
            continue
        
        # Store all feature scores for detailed analysis
        analysis['feature_scores'][feature] = float(value)
        
        # Determine feature category
        feature_category = None
        for category, keywords in feature_categories.items():
            if any(keyword in feature.lower() for keyword in keywords):
                feature_category = category
                break
        
        # Check high risk
        for risk_feature, threshold in high_risk_features.items():
            if risk_feature in feature.lower():
                # Handle inverted scales (lower is worse)
                is_inverted = risk_feature in ['work_life_balance', 'job_satisfaction', 'sleep', 'physical_health', 'mental_health']
                
                if (not is_inverted and value >= threshold) or (is_inverted and value <= threshold):
                    analysis['high_risk_indicators'].append({
                        "feature": feature,
                        "value": float(value),
                        "threshold": threshold,
                        "description": f"{'Low' if is_inverted else 'High'} {feature.replace('_', ' ')}",
                        "category": feature_category or "general",
                        "severity": "high"
                    })
        
        # Check moderate risk
        for risk_feature, threshold in moderate_risk_features.items():
            if risk_feature in feature.lower():
                is_inverted = risk_feature in ['sleep', 'physical_health', 'mental_health']
                
                if (not is_inverted and value >= threshold) or (is_inverted and value <= threshold):
                    # Don't duplicate if already in high risk
                    if not any(indicator['feature'] == feature for indicator in analysis['high_risk_indicators']):
                        analysis['moderate_risk_indicators'].append({
                            "feature": feature,
                            "value": float(value),
                            "threshold": threshold,
                            "description": f"Elevated {feature.replace('_', ' ')}",
                            "category": feature_category or "general",
                            "severity": "moderate"
                        })
        
        # Check protective factors
        for protective_feature, threshold in protective_features.items():
            if protective_feature in feature.lower() and value >= threshold:
                analysis['protective_factors'].append({
                    "feature": feature,
                    "value": float(value),
                    "description": f"Strong {feature.replace('_', ' ')}",
                    "category": feature_category or "general"
                })
    
    return analysis


def generate_confusion_matrix_image(metadata: Dict) -> Optional[str]:
    """Generate confusion matrix visualization from training history."""
    try:
        if 'training_metrics' not in metadata:
            return None
        
        metrics = metadata['training_metrics']
        if 'confusion_matrix' not in metrics:
            return None
        
        cm = np.array(metrics['confusion_matrix'])
        classes = metrics.get('classes', ['Low', 'Moderate', 'High'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Model Confusion Matrix\n(From Training Data)', fontsize=14, pad=20)
        
        # Add accuracy text
        accuracy = metrics.get('test_accuracy', 0)
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
                ha='center', transform=ax.transAxes, fontsize=11)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix: {e}")
        return None


def generate_detailed_model_info(metadata: Dict) -> Dict[str, Any]:
    """Generate comprehensive model information section."""
    
    model_info = {
        "model_version": {
            "version": metadata.get('version', 'latest'),
            "model_type": metadata.get('best_model') or metadata.get('model_type', 'Unknown'),
            "trained_at": metadata.get('training_date'),
            "status": "active"
        },
        
        "training_data": {
            "total_records": metadata.get('records_used', 0),
            "training_samples": metadata.get('n_train_samples', 0),
            "testing_samples": metadata.get('n_test_samples', 0),
            "number_of_features": metadata.get('n_features', 0),
            "class_distribution": metadata.get('class_distribution', {}),
            "data_split": {
                "train_percentage": round((metadata.get('n_train_samples', 0) / metadata.get('records_used', 1)) * 100, 1),
                "test_percentage": round((metadata.get('n_test_samples', 0) / metadata.get('records_used', 1)) * 100, 1)
            }
        },
        
        "model_performance": {
            "overall_accuracy": round(metadata.get('training_metrics', {}).get('accuracy', 0), 2),
            "balanced_accuracy": round(metadata.get('training_metrics', {}).get('balanced_accuracy', 0), 2),
            "cohen_kappa": round(metadata.get('training_metrics', {}).get('cohen_kappa', 0), 4),
            "matthews_correlation": round(metadata.get('training_metrics', {}).get('matthews_corrcoef', 0), 4),
            "weighted_metrics": {
                "precision": round(metadata.get('training_metrics', {}).get('weighted_precision', 0), 2),
                "recall": round(metadata.get('training_metrics', {}).get('weighted_recall', 0), 2),
                "f1_score": round(metadata.get('training_metrics', {}).get('weighted_f1', 0), 2)
            }
        },
        
        "per_class_performance": {},
        
        "cross_validation": {},
        
        "model_comparison": metadata.get('model_comparison', {}),
        
        "interpretation": {
            "accuracy_meaning": _interpret_accuracy(metadata.get('training_metrics', {}).get('accuracy', 0)),
            "reliability": _interpret_reliability(
                metadata.get('training_metrics', {}).get('cohen_kappa', 0),
                metadata.get('training_metrics', {}).get('matthews_corrcoef', 0)
            ),
            "data_quality": _interpret_data_quality(
                metadata.get('records_used', 0),
                metadata.get('class_distribution', {})
            )
        }
    }
    
    # Add per-class performance
    per_class = metadata.get('training_metrics', {}).get('per_class', {})
    for class_name, metrics in per_class.items():
        model_info['per_class_performance'][class_name] = {
            "precision": round(metrics.get('precision', 0), 2),
            "recall": round(metrics.get('recall', 0), 2),
            "f1_score": round(metrics.get('f1_score', 0), 2),
            "support": metrics.get('support', 0),
            "interpretation": _interpret_class_performance(
                class_name,
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0)
            )
        }
    
    # Add cross-validation scores
    cv_scores = metadata.get('cv_scores', {})
    for model_name, scores in cv_scores.items():
        model_info['cross_validation'][model_name] = {
            "mean_accuracy": round(scores.get('mean', 0), 2),
            "std_deviation": round(scores.get('std', 0), 2),
            "individual_scores": [round(s * 100, 2) for s in scores.get('scores', [])],
            "consistency": _interpret_cv_consistency(scores.get('std', 0))
        }
    
    return model_info


def _interpret_accuracy(accuracy: float) -> str:
    """Interpret what the accuracy means."""
    if accuracy >= 95:
        return f"Exceptional accuracy ({accuracy:.1f}%) - The model correctly predicts burnout level in nearly all cases."
    elif accuracy >= 90:
        return f"Excellent accuracy ({accuracy:.1f}%) - The model is highly reliable for burnout prediction."
    elif accuracy >= 85:
        return f"Very good accuracy ({accuracy:.1f}%) - The model provides dependable predictions in most cases."
    elif accuracy >= 80:
        return f"Good accuracy ({accuracy:.1f}%) - The model is generally reliable but may occasionally misclassify."
    else:
        return f"Moderate accuracy ({accuracy:.1f}%) - Predictions should be considered alongside other assessments."


def _interpret_reliability(kappa: float, mcc: float) -> str:
    """Interpret reliability metrics."""
    avg_reliability = (kappa + mcc) / 2
    
    if avg_reliability >= 0.9:
        return "Exceptional reliability - Agreement far exceeds chance, indicating very trustworthy predictions."
    elif avg_reliability >= 0.8:
        return "Strong reliability - High agreement beyond chance, predictions are highly dependable."
    elif avg_reliability >= 0.6:
        return "Good reliability - Substantial agreement beyond chance, predictions are generally trustworthy."
    else:
        return "Moderate reliability - Fair agreement beyond chance, consider multiple assessment methods."


def _interpret_data_quality(records: int, distribution: Dict) -> str:
    """Interpret data quality based on size and balance."""
    if records >= 500:
        size_quality = "large, robust dataset"
    elif records >= 300:
        size_quality = "adequately sized dataset"
    elif records >= 100:
        size_quality = "moderate dataset"
    else:
        size_quality = "small dataset"
    
    # Check class balance
    if distribution:
        values = list(distribution.values())
        max_val = max(values)
        min_val = min(values)
        imbalance_ratio = max_val / min_val if min_val > 0 else float('inf')
        
        if imbalance_ratio <= 1.5:
            balance = "well-balanced classes"
        elif imbalance_ratio <= 2.5:
            balance = "reasonably balanced classes"
        else:
            balance = "imbalanced classes"
        
        return f"Trained on {size_quality} with {balance}."
    
    return f"Trained on {size_quality}."


def _interpret_class_performance(class_name: str, precision: float, recall: float, f1: float) -> str:
    """Interpret per-class performance."""
    if f1 >= 90:
        return f"Excellent at identifying {class_name} burnout cases."
    elif f1 >= 85:
        return f"Very good at identifying {class_name} burnout cases."
    elif f1 >= 80:
        return f"Good at identifying {class_name} burnout cases."
    else:
        return f"Moderately reliable for {class_name} burnout cases."


def _interpret_cv_consistency(std: float) -> str:
    """Interpret cross-validation consistency."""
    if std <= 2:
        return "Very consistent across different data splits"
    elif std <= 4:
        return "Reasonably consistent across different data splits"
    else:
        return "Some variability across different data splits"


def generate_actionable_insights(
    input_analysis: Dict[str, Any],
    burnout_level: str,
    feature_importance: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate detailed, actionable insights based on what's driving the prediction."""
    
    insights = {
        "primary_drivers": [],
        "contributing_factors": [],
        "areas_needing_attention": [],
        "strengths_to_maintain": [],
        "detailed_recommendations": []
    }
    
    # Categorize issues by domain
    domain_issues = {}
    
    for indicator in input_analysis.get('high_risk_indicators', []):
        category = indicator.get('category', 'general')
        if category not in domain_issues:
            domain_issues[category] = {
                'high_risk': [],
                'moderate_risk': [],
                'count': 0
            }
        domain_issues[category]['high_risk'].append(indicator)
        domain_issues[category]['count'] += 2  # High risk counts more
    
    for indicator in input_analysis.get('moderate_risk_indicators', []):
        category = indicator.get('category', 'general')
        if category not in domain_issues:
            domain_issues[category] = {
                'high_risk': [],
                'moderate_risk': [],
                'count': 0
            }
        domain_issues[category]['moderate_risk'].append(indicator)
        domain_issues[category]['count'] += 1
    
    # Sort domains by severity
    sorted_domains = sorted(domain_issues.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Generate primary drivers (top 3 domains)
    for domain, issues in sorted_domains[:3]:
        driver = {
            "domain": domain.replace('_', ' ').title(),
            "severity": "critical" if issues['high_risk'] else "moderate",
            "affected_features": [],
            "explanation": "",
            "why_it_matters": "",
            "specific_actions": []
        }
        
        # Add all affected features
        for indicator in issues['high_risk'] + issues['moderate_risk']:
            driver['affected_features'].append({
                "feature": indicator['feature'].replace('_', ' ').title(),
                "your_score": indicator['value'],
                "concern_threshold": indicator['threshold'],
                "status": "critical" if indicator in issues['high_risk'] else "warning"
            })
        
        # Generate domain-specific insights
        domain_insights = _generate_domain_specific_insights(domain, issues, burnout_level)
        driver['explanation'] = domain_insights['explanation']
        driver['why_it_matters'] = domain_insights['why_it_matters']
        driver['specific_actions'] = domain_insights['actions']
        
        insights['primary_drivers'].append(driver)
    
    # Generate detailed recommendations per domain
    for domain, issues in sorted_domains:
        if issues['high_risk'] or issues['moderate_risk']:
            detailed_rec = _generate_detailed_domain_recommendations(domain, issues, burnout_level)
            insights['detailed_recommendations'].append(detailed_rec)
    
    # Add protective factors
    for factor in input_analysis.get('protective_factors', [])[:5]:
        insights['strengths_to_maintain'].append({
            "factor": factor['feature'].replace('_', ' ').title(),
            "score": factor['value'],
            "note": f"This is a strength - continue maintaining this positive aspect."
        })
    
    return insights


def _generate_domain_specific_insights(domain: str, issues: Dict, burnout_level: str) -> Dict[str, Any]:
    """Generate specific insights for each domain."""
    
    domain_knowledge = {
        'sleep': {
            'explanation': (
                "Your sleep patterns are significantly impacting your burnout risk. "
                "Poor sleep directly affects emotional regulation, cognitive function, and stress resilience. "
                f"With {len(issues['high_risk'])} critical sleep-related indicators, this is a primary factor."
            ),
            'why_it_matters': (
                "Sleep deprivation amplifies stress responses, reduces problem-solving abilities, "
                "and increases emotional reactivity - all of which accelerate burnout progression."
            ),
            'actions': [
                "Establish consistent sleep schedule (same bedtime/wake time daily)",
                "Create 30-minute wind-down routine before bed",
                "Eliminate screens 1 hour before sleep",
                "Keep bedroom cool (60-67°F/15-19°C) and dark",
                "Limit caffeine after 2 PM",
                "If sleep problems persist >2 weeks, consult healthcare provider"
            ]
        },
        'physical_health': {
            'explanation': (
                "Your physical health scores indicate that bodily stress and fatigue are contributing "
                "significantly to your burnout. Physical symptoms often manifest before psychological awareness."
            ),
            'why_it_matters': (
                "Physical health problems create a stress cycle - poor health increases mental stress, "
                "which further deteriorates physical health. Breaking this cycle is crucial."
            ),
            'actions': [
                "Schedule medical check-up to rule out underlying conditions",
                "Start with 10-minute daily walks (gradually increase)",
                "Stay hydrated (8 glasses water daily)",
                "Eat regular meals with protein and vegetables",
                "Take 5-minute movement breaks every hour",
                "Consider physical therapy if experiencing chronic pain"
            ]
        },
        'mental_health': {
            'explanation': (
                "Your mental health indicators show significant emotional distress that is driving your burnout risk. "
                "This includes anxiety, depression, or emotional exhaustion patterns."
            ),
            'why_it_matters': (
                "Mental health symptoms both contribute to and result from burnout. "
                "Without intervention, this can create a downward spiral affecting all life areas."
            ),
            'actions': [
                "Schedule appointment with mental health professional within 1 week",
                "Use crisis helpline if experiencing thoughts of self-harm",
                "Practice daily grounding exercises (5-4-3-2-1 technique)",
                "Reach out to trusted friend/family member for support",
                "Consider evidence-based apps (Headspace, Calm, Sanvello)",
                "Explore workplace mental health resources or EAP"
            ]
        },
        'workload': {
            'explanation': (
                "Your workload levels are unsustainable and are a primary driver of your burnout. "
                "The volume or intensity of work exceeds your capacity to recover."
            ),
            'why_it_matters': (
                "Chronic overwork depletes mental and physical resources faster than they can be replenished, "
                "leading to decreased productivity and eventual breakdown."
            ),
            'actions': [
                "Document current workload and time spent on each task",
                "Schedule meeting with supervisor to discuss workload concerns",
                "Identify 3 tasks to delegate, defer, or eliminate",
                "Set firm boundaries on work hours (no emails after 6 PM)",
                "Learn to say 'no' to non-essential requests",
                "Use time-blocking to protect focused work periods"
            ]
        },
        'social_support': {
            'explanation': (
                "Lack of social support is leaving you without crucial emotional resources to buffer stress. "
                "Social isolation amplifies burnout symptoms."
            ),
            'why_it_matters': (
                "Strong social connections are one of the most protective factors against burnout. "
                "They provide emotional validation, practical help, and stress relief."
            ),
            'actions': [
                "Reach out to 1-2 people you trust this week",
                "Join a support group (in-person or online)",
                "Schedule regular coffee/lunch with colleagues or friends",
                "Consider peer support programs at work/school",
                "Participate in group activities aligned with your interests",
                "If feeling isolated, start with online communities"
            ]
        },
        'work_life_balance': {
            'explanation': (
                "Your work-life balance is severely compromised, with work demands encroaching on personal time "
                "and recovery periods."
            ),
            'why_it_matters': (
                "Without adequate recovery time, your body and mind cannot restore resources needed for sustained performance. "
                "This imbalance is unsustainable long-term."
            ),
            'actions': [
                "Establish 'hard stop' time for work each day",
                "Create buffer zones between work and personal time",
                "Schedule non-negotiable personal activities weekly",
                "Use 'out of office' features genuinely",
                "Identify activities that truly recharge you",
                "Consider flexible work arrangements if available"
            ]
        },
        'stress': {
            'explanation': (
                "Your stress levels are chronically elevated, keeping your body in persistent fight-or-flight mode. "
                "This is depleting your adaptive capacity."
            ),
            'why_it_matters': (
                "Chronic stress rewires your brain, impairs immune function, and accelerates aging. "
                "It's not just feeling stressed - it's physiological damage accumulating."
            ),
            'actions': [
                "Practice daily stress-reduction technique (deep breathing, meditation)",
                "Identify top 3 stressors and create action plan for each",
                "Use stress tracking app to identify patterns",
                "Take regular breaks throughout day (Pomodoro technique)",
                "Engage in physical activity to discharge stress hormones",
                "Consider professional stress management coaching"
            ]
        },
        'job_satisfaction': {
            'explanation': (
                "Low job satisfaction indicates a fundamental mismatch between your needs/values and your work reality. "
                "This dissatisfaction is eroding your motivation and engagement."
            ),
            'why_it_matters': (
                "When work feels meaningless or unrewarding, every task becomes a burden. "
                "This chronic dissatisfaction is a major burnout predictor."
            ),
            'actions': [
                "Identify specific aspects of work that are unsatisfying",
                "Explore opportunities to align work with your strengths",
                "Discuss role modifications with supervisor",
                "Seek projects that align with your values",
                "Consider whether this role/field is right long-term",
                "Explore internal transfers or role changes"
            ]
        }
    }
    
    # Return domain-specific insights or generic ones
    if domain in domain_knowledge:
        return domain_knowledge[domain]
    else:
        return {
            'explanation': f"Your {domain.replace('_', ' ')} scores indicate this area needs attention.",
            'why_it_matters': "This factor is contributing to your overall burnout risk.",
            'actions': [
                f"Assess your {domain.replace('_', ' ')} patterns",
                "Identify specific problems in this area",
                "Create action plan to improve",
                "Monitor progress weekly"
            ]
        }


def _generate_detailed_domain_recommendations(domain: str, issues: Dict, burnout_level: str) -> Dict[str, Any]:
    """Generate comprehensive recommendations for each problem domain."""
    
    recommendation = {
        "domain": domain.replace('_', ' ').title(),
        "severity_level": "critical" if issues['high_risk'] else "moderate",
        "immediate_steps": [],
        "short_term_goals": [],
        "long_term_strategies": [],
        "resources": [],
        "success_metrics": []
    }
    
    # Domain-specific recommendations
    recommendations_db = {
        'sleep': {
            'immediate': [
                "Tonight: Set alarm for consistent wake time",
                "Remove electronic devices from bedroom",
                "Take warm shower 1 hour before bed"
            ],
            'short_term': [
                "Maintain sleep schedule for 2 weeks (including weekends)",
                "Track sleep quality daily",
                "Identify and eliminate sleep disruptors"
            ],
            'long_term': [
                "Develop permanent sleep hygiene routine",
                "Address underlying sleep disorders if identified",
                "Maintain 7-9 hours sleep consistently"
            ],
            'resources': [
                "Sleep diary app (Sleep Cycle, Sleep as Android)",
                "CBT-I (Cognitive Behavioral Therapy for Insomnia)",
                "Sleep Foundation website (sleepfoundation.org)",
                "Consult sleep specialist if needed"
            ],
            'metrics': [
                "Hours of sleep per night (target: 7-9)",
                "Sleep onset time (target: <20 minutes)",
                "Number of awakenings (target: <2)",
                "Morning energy level (rate 1-10, target: >7)"
            ]
        },
        'physical_health': {
            'immediate': [
                "Schedule doctor appointment this week",
                "Take 10-minute walk today",
                "Drink 4 glasses of water before lunch"
            ],
            'short_term': [
                "Complete medical evaluation",
                "Establish 30-minute daily movement routine",
                "Improve nutrition (more vegetables, less processed food)"
            ],
            'long_term': [
                "Maintain regular exercise 3-5x per week",
                "Annual health check-ups",
                "Sustainable healthy eating habits"
            ],
            'resources': [
                "Fitness tracking app (MyFitnessPal, Fitbit)",
                "Online workout videos (YouTube, fitness apps)",
                "Nutrition guidance (registered dietitian)",
                "Physical therapy if needed"
            ],
            'metrics': [
                "Days of physical activity per week (target: 5)",
                "Energy levels throughout day (rate 1-10)",
                "Sick days taken (track decrease)",
                "Physical symptoms frequency (headaches, etc.)"
            ]
        },
        'mental_health': {
            'immediate': [
                "Call counseling center/EAP today",
                "Use grounding technique if feeling overwhelmed",
                "Reach out to one supportive person"
            ],
            'short_term': [
                "Attend first therapy session within 2 weeks",
                "Practice daily mindfulness (10 minutes)",
                "Journal emotions and triggers"
            ],
            'long_term': [
                "Continue therapy as recommended",
                "Build emotional regulation skills",
                "Develop relapse prevention plan"
            ],
            'resources': [
                "Crisis hotline: 988 (US Suicide & Crisis Lifeline)",
                "BetterHelp or Talkspace (online therapy)",
                "Mental health apps (Sanvello, Headspace)",
                "Campus counseling center or EAP"
            ],
            'metrics': [
                "Therapy sessions attended",
                "PHQ-9/GAD-7 scores (track monthly)",
                "Good mental health days per week",
                "Coping skills used vs. avoidance behaviors"
            ]
        },
        'workload': {
            'immediate': [
                "List all current commitments and deadlines",
                "Identify 1 task to delegate or defer today",
                "Block 'no meeting' time on calendar"
            ],
            'short_term': [
                "Meet with supervisor about workload concerns",
                "Renegotiate 2-3 major deadlines",
                "Establish clear priorities"
            ],
            'long_term': [
                "Maintain sustainable workload through boundaries",
                "Regular workload reviews with supervisor",
                "Develop time management systems"
            ],
            'resources': [
                "Time tracking tools (Toggl, RescueTime)",
                "Project management apps (Todoist, Asana)",
                "Time management courses",
                "Coaching or mentoring support"
            ],
            'metrics': [
                "Hours worked per week (target: <45)",
                "Tasks completed vs. new tasks added",
                "Stress level at end of workday (rate 1-10)",
                "Number of declined requests"
            ]
        }
    }
    
    # Get recommendations or use generic
    if domain in recommendations_db:
        domain_recs = recommendations_db[domain]
        recommendation['immediate_steps'] = domain_recs['immediate']
        recommendation['short_term_goals'] = domain_recs['short_term']
        recommendation['long_term_strategies'] = domain_recs['long_term']
        recommendation['resources'] = domain_recs['resources']
        recommendation['success_metrics'] = domain_recs['metrics']
    
    return recommendation


def generate_detailed_report(
    result: Dict[str, Any],
    input_data: Dict[str, Any],
    preprocessor: Dict,
    model: BaseEstimator,
    metadata: Dict
) -> Dict[str, Any]:
    """Generate comprehensive analysis report."""
    
    # Generate model info first
    model_info_detailed = generate_detailed_model_info(metadata)
    
    report = {
        "prediction_summary": {
            "burnout_level": result['prediction'],
            "confidence": result['confidence'],
            "risk_category": _get_risk_category(result['prediction']),
        },
        
        "model_information": model_info_detailed,
        
        "how_prediction_works": {
            "explanation": (
                f"This prediction uses a {metadata.get('best_model') or metadata.get('model_type', 'machine learning')} model "
                f"trained on {metadata.get('records_used', 0)} real burnout assessments. "
                f"The model analyzes {metadata.get('n_features', len(preprocessor['feature_names']))} "
                f"different features from your responses to identify patterns associated with different burnout levels."
            ),
            "steps": [
                {
                    "step": 1,
                    "name": "Data Normalization",
                    "description": "Your survey responses are converted to numerical values and standardized."
                },
                {
                    "step": 2,
                    "name": "Feature Processing",
                    "description": "Missing values are handled and features are scaled to ensure fair comparison."
                },
                {
                    "step": 3,
                    "name": "Pattern Recognition",
                    "description": f"The model (trained on {metadata.get('records_used', 0)} cases) identifies patterns matching known burnout indicators."
                },
                {
                    "step": 4,
                    "name": "Classification",
                    "description": f"Based on these patterns, the model assigns a burnout level with {result['confidence']:.1f}% confidence."
                }
            ]
        },
        
        "feature_analysis": None,  # Will be filled below
        "input_analysis": None,    # Will be filled below
        "actionable_insights": None,  # Will be filled below
        "confusion_matrix_image": None,  # Will be filled below
        
        "clinical_interpretation": _generate_clinical_interpretation(
            result['prediction'],
            result['confidence']
        ),
        
        "what_this_means": _generate_what_this_means(
            result['prediction'],
            result['probabilities']
        ),
    }
    
    # Add feature importance
    feature_importance = get_feature_importance(model, preprocessor['feature_names'])
    if feature_importance:
        report['feature_analysis'] = {
            "description": "These features had the most influence on the model's predictions during training:",
            "top_features": feature_importance
        }
    
    # Add input analysis
    input_analysis = analyze_input_features(input_data, preprocessor)
    report['input_analysis'] = {
        "description": "Analysis of your specific responses:",
        "details": input_analysis
    }
    
    # Add actionable insights
    actionable_insights = generate_actionable_insights(
        input_analysis,
        result['prediction'],
        feature_importance
    )
    report['actionable_insights'] = actionable_insights
    
    # Add confusion matrix
    cm_image = generate_confusion_matrix_image(metadata)
    if cm_image:
        report['confusion_matrix_image'] = cm_image
    
    return report


def _get_risk_category(burnout_level: str) -> str:
    """Convert burnout level to risk category."""
    mapping = {
        "High": "Critical",
        "Moderate": "Warning",
        "Low": "Healthy"
    }
    return mapping.get(burnout_level, "Unknown")


def _generate_clinical_interpretation(burnout_level: str, confidence: float) -> Dict[str, Any]:
    """Generate clinical interpretation of results."""
    
    interpretations = {
        "High": {
            "severity": "Severe",
            "description": (
                "Your responses indicate significant burnout symptoms that require immediate attention. "
                "You are likely experiencing chronic emotional exhaustion, detachment from work/studies, "
                "and reduced sense of accomplishment."
            ),
            "symptoms": [
                "Persistent emotional exhaustion",
                "Cynicism and detachment",
                "Reduced professional efficacy",
                "Physical symptoms (headaches, sleep problems)",
                "Cognitive difficulties (concentration, memory)"
            ],
            "urgency": "Immediate intervention recommended"
        },
        "Moderate": {
            "severity": "Moderate",
            "description": (
                "Your responses show warning signs of developing burnout. You are experiencing notable "
                "stress levels that, if unaddressed, could progress to more severe burnout."
            ),
            "symptoms": [
                "Increased fatigue and irritability",
                "Growing sense of ineffectiveness",
                "Decreased motivation",
                "Early signs of emotional exhaustion",
                "Work-life balance concerns"
            ],
            "urgency": "Preventive action recommended"
        },
        "Low": {
            "severity": "Minimal",
            "description": (
                "Your responses indicate you are managing stress relatively well. While everyone experiences "
                "some stress, your levels are within a healthy range."
            ),
            "symptoms": [
                "Normal stress levels",
                "Adequate coping mechanisms",
                "Maintained work engagement",
                "Healthy work-life balance"
            ],
            "urgency": "Continue healthy practices"
        }
    }
    
    interp = interpretations.get(burnout_level, interpretations["Low"])
    interp['confidence_note'] = (
        f"The model is {confidence:.1f}% confident in this assessment. "
        f"{'This high confidence suggests clear patterns in your responses.' if confidence >= 80 else 'Consider retaking the assessment if you feel this does not reflect your situation.'}"
    )
    
    return interp


def _generate_what_this_means(burnout_level: str, probabilities: Dict[str, float]) -> Dict[str, Any]:
    """Generate explanation of what the prediction means."""
    
    probs_sorted = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_class, top_prob = probs_sorted[0]
    second_class, second_prob = probs_sorted[1] if len(probs_sorted) > 1 else (None, 0)
    
    explanation = {
        "primary_classification": {
            "level": burnout_level,
            "probability": top_prob,
            "meaning": f"The model found your response pattern most closely matches {burnout_level} burnout."
        }
    }
    
    if second_prob > 20:
        explanation["alternative_consideration"] = {
            "level": second_class,
            "probability": second_prob,
            "meaning": (
                f"There is also a {second_prob:.1f}% chance you may be experiencing {second_class} burnout. "
                f"This suggests you may be in a transitional state or showing mixed symptoms."
            )
        }
    
    # Add probability distribution explanation
    explanation["probability_distribution"] = {
        "explanation": (
            "The percentages show how strongly your responses match each burnout category. "
            "Higher percentages indicate stronger pattern matches with that category."
        ),
        "breakdown": {
            cls: {
                "percentage": prob,
                "interpretation": _interpret_probability(prob)
            }
            for cls, prob in probabilities.items()
        }
    }
    
    return explanation


def _interpret_probability(prob: float) -> str:
    """Interpret what a probability percentage means."""
    if prob >= 70:
        return "Strong match - your responses clearly align with this category"
    elif prob >= 50:
        return "Moderate match - significant patterns align with this category"
    elif prob >= 30:
        return "Possible match - some patterns suggest this category"
    else:
        return "Low match - few patterns align with this category"


def predict_burnout_simple(input_data: Dict[str, Any], model: BaseEstimator, preprocessor: Dict) -> Dict[str, Any]:
    """
    Simple prediction logic matching test_prediction.py EXACTLY.
    
    Args:
        input_data: Normalized survey responses
        model: Loaded sklearn model
        preprocessor: Loaded preprocessor dict
        
    Returns:
        Dict with prediction, confidence, and probabilities
    """
    try:
        # Get feature names from preprocessor
        feature_names = preprocessor['feature_names']
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([input_data])
        
        # Add missing features with NaN
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = np.nan
        
        # Select and order features correctly
        df = df[feature_names]
        
        # Convert everything to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Apply preprocessing for categorical columns
        for col in preprocessor['categorical_columns']:
            if col in feature_names and col in preprocessor['label_encoders']:
                le = preprocessor['label_encoders'][col]
                col_position = list(feature_names).index(col)
                df.iloc[:, col_position] = le.transform(
                    df.iloc[:, col_position].astype(str).fillna('unknown')
                )
        
        # Convert to numpy array (eliminates feature name warnings)
        X = df.to_numpy()
        
        # Impute and scale using numpy arrays
        X_imputed = preprocessor['imputer'].transform(X)
        X_scaled = preprocessor['scaler'].transform(X_imputed)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Get class names
        classes = model.classes_
        
        # Build result
        result = {
            'prediction': str(prediction),
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


def generate_enhanced_recommendations(
    burnout_level: str,
    confidence: float,
    input_analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate enhanced recommendations based on burnout level and specific indicators."""
    recommendations = []
    
    # Base recommendations
    if burnout_level == "High":
        recommendations = [
            {
                "category": "Immediate Action",
                "priority": "critical",
                "suggestion": "Schedule appointment with mental health professional within 48-72 hours",
                "rationale": "Severe burnout requires professional intervention",
                "resources": [
                    "Campus counseling center",
                    "Employee assistance program",
                    "Mental health hotlines"
                ]
            },
            {
                "category": "Emergency Self-Care",
                "priority": "high",
                "suggestion": "Prioritize 7-9 hours of sleep, cancel non-essential commitments",
                "rationale": "Immediate rest is needed to prevent complete breakdown",
                "action_items": [
                    "Block out recovery time in calendar",
                    "Communicate with supervisor/professor about reduced capacity",
                    "Identify 3 activities to postpone or delegate"
                ]
            },
            {
                "category": "Workload Management",
                "priority": "high",
                "suggestion": "Request meeting with supervisor/advisor to discuss workload reduction",
                "rationale": "Continuing at current pace will worsen symptoms",
                "talking_points": [
                    "Specific symptoms you're experiencing",
                    "Timeline for when symptoms began",
                    "Request for temporary workload adjustment"
                ]
            }
        ]
    elif burnout_level == "Moderate":
        recommendations = [
            {
                "category": "Preventive Assessment",
                "priority": "high",
                "suggestion": "Schedule consultation with counselor within 1-2 weeks",
                "rationale": "Early intervention prevents progression to severe burnout",
                "benefits": [
                    "Learn personalized coping strategies",
                    "Identify specific stressors",
                    "Develop action plan"
                ]
            },
            {
                "category": "Life Balance",
                "priority": "normal",
                "suggestion": "Conduct time audit and identify 2-3 commitments to reduce",
                "rationale": "Rebalancing now prevents future crisis",
                "action_items": [
                    "Track time for one week",
                    "Identify energy drains",
                    "Practice saying 'no' to new requests"
                ]
            },
            {
                "category": "Stress Management",
                "priority": "normal",
                "suggestion": "Begin daily 10-15 minute relaxation practice",
                "rationale": "Regular practice builds resilience",
                "techniques": [
                    "Deep breathing exercises",
                    "Progressive muscle relaxation",
                    "Mindfulness meditation",
                    "Guided imagery"
                ]
            }
        ]
    else:  # Low
        recommendations = [
            {
                "category": "Maintenance",
                "priority": "normal",
                "suggestion": "Continue current wellness practices",
                "rationale": "Your current strategies are working well",
                "reinforcement": "Keep doing what you're doing!"
            },
            {
                "category": "Monitoring",
                "priority": "normal",
                "suggestion": "Monthly self-assessment to detect early warning signs",
                "rationale": "Regular monitoring enables timely intervention",
                "warning_signs": [
                    "Increased irritability",
                    "Sleep changes",
                    "Loss of motivation",
                    "Physical symptoms"
                ]
            }
        ]
    
    # Add specific recommendations based on high-risk indicators
    if input_analysis and input_analysis.get('high_risk_indicators'):
        for indicator in input_analysis['high_risk_indicators'][:3]:  # Top 3
            feature = indicator['feature'].replace('_', ' ').title()
            recommendations.append({
                "category": f"Address {feature}",
                "priority": "high",
                "suggestion": f"Specific interventions needed for high {feature.lower()}",
                "rationale": f"Your {feature.lower()} score indicates this is a critical area",
                "target": indicator['value']
            })
    
    return recommendations


def generate_simple_interpretation(burnout_level: str, confidence: float, probabilities: Dict[str, float]) -> str:
    """Generate simple interpretation of results."""
    interpretation = f"Prediction: {burnout_level} Burnout (Confidence: {confidence:.1f}%)\n\n"
    
    if burnout_level == "High":
        interpretation += ("Your assessment indicates significant burnout symptoms. Multiple factors including "
                          "emotional exhaustion and stress are contributing to this classification. "
                          "Professional support is recommended.")
    elif burnout_level == "Moderate":
        interpretation += ("Your assessment shows notable warning signs of burnout. You are experiencing "
                          "considerable stress that indicates developing symptoms. "
                          "Early intervention can prevent progression.")
    else:
        interpretation += ("Your assessment indicates you are managing stress relatively well. "
                          "Continue maintaining healthy boundaries and stress management practices.")
    
    interpretation += "\n\nProbability Distribution:\n"
    for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob / 5)
        interpretation += f"  {cls:12} {prob:6.2f}% {bar}\n"
    
    return interpretation


def run_prediction(
    payload: Dict[str, Any],
    log_to_firestore: bool = True,
    user_context: Optional[Dict[str, Any]] = None,
    include_detailed_report: bool = True,
) -> Dict[str, Any]:
    """
    Run simplified burnout prediction with detailed analysis.
    
    Args:
        payload: Raw survey responses (dict with question keys and answers)
        log_to_firestore: Whether to log prediction to database
        user_context: Optional metadata (user_id, session_id, etc.)
        include_detailed_report: Whether to include detailed analysis report
        
    Returns:
        Dictionary containing prediction results and detailed report
    """
    try:
        start_time = datetime.utcnow()
        
        # Step 1: Load model
        logger.info("Loading model and preprocessor...")
        model, preprocessor, metadata = load_model_and_preprocessor()
        
        if model is None or preprocessor is None:
            raise RuntimeError("No model available. Please train a model first.")
        
        logger.info(f"Model loaded: {metadata.get('model_type')}")
        
        # Step 2: Normalize input
        logger.info(f"Normalizing {len(payload)} input features...")
        normalized_data = normalize_input(payload)
        logger.info(f"Normalized to {len(normalized_data)} features")
        
        # Step 3: Run prediction
        logger.info("Running prediction...")
        result = predict_burnout_simple(normalized_data, model, preprocessor)
        
        logger.info(f"Prediction: {result['prediction']} ({result['confidence']:.1f}% confidence)")
        
        # Step 4: Generate detailed report
        detailed_report = None
        if include_detailed_report:
            logger.info("Generating detailed analysis report...")
            detailed_report = generate_detailed_report(
                result, normalized_data, preprocessor, model, metadata
            )
        
        # Step 5: Generate enhanced recommendations
        input_analysis = detailed_report['input_analysis']['details'] if detailed_report else None
        recommendations = generate_enhanced_recommendations(
            result['prediction'],
            result['confidence'],
            input_analysis
        )
        
        # Step 6: Generate interpretation
        interpretation = generate_simple_interpretation(
            result['prediction'],
            result['confidence'],
            result['probabilities']
        )
        
        # Step 7: Build response
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = {
            "success": True,
            "burnout_level": result['prediction'],
            "probability": round(result['confidence'] / 100, 4),
            "confidence_score": round(result['confidence'], 2),
            
            "all_probabilities": {
                cls: round(prob / 100, 4)
                for cls, prob in result['probabilities'].items()
            },
            
            "interpretation": interpretation,
            "recommendations": recommendations,
            
            "probability_breakdown": {
                cls: {
                    "probability": round(prob / 100, 4),
                    "percentage": round(prob, 2),
                    "confidence_level": _get_confidence_level(prob),
                }
                for cls, prob in result['probabilities'].items()
            },
            
            "model_info": {
                "version": metadata.get("version"),
                "model_type": metadata.get("model_type"),
                "source": metadata.get("source"),
            },
            
            "prediction_metadata": {
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "features_used": len(preprocessor['feature_names']),
                "features_provided": len(payload),
                "processing_time_seconds": round(processing_time, 3),
            }
        }
        
        # Add detailed report if requested
        if detailed_report:
            response['detailed_report'] = detailed_report
        
        # Step 8: Log to Firestore (optional)
        if log_to_firestore and db:
            try:
                _log_to_firestore(response, normalized_data, user_context)
            except Exception as e:
                logger.warning(f"Firestore logging failed: {e}")
        
        logger.info(f"Prediction completed in {processing_time:.3f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "message": "Prediction failed. Please check your input and try again."
        }


def _get_confidence_level(percentage: float) -> str:
    """Convert percentage to confidence level."""
    if percentage >= 90:
        return "Exceptional"
    elif percentage >= 85:
        return "Very High"
    elif percentage >= 70:
        return "High"
    elif percentage >= 55:
        return "Moderate"
    else:
        return "Low"


def _log_to_firestore(
    response: Dict[str, Any],
    normalized_data: Dict[str, Any],
    user_context: Optional[Dict[str, Any]]
):
    """Log prediction to Firestore."""
    if db is None:
        return
    
    try:
        doc = {
            "prediction": response["burnout_level"],
            "confidence": response["confidence_score"],
            "all_probabilities": convert_to_native_types(response["all_probabilities"]),
            "recommendations": convert_to_native_types(response["recommendations"]),
            "interpretation": response["interpretation"],
            "normalized_data": convert_to_native_types(normalized_data),
            "model_info": convert_to_native_types(response["model_info"]),
            "processing_metadata": convert_to_native_types(response["prediction_metadata"]),
            "user_context": convert_to_native_types(user_context or {}),
            "timestamp": datetime.utcnow(),
            "status": "completed",
        }
        
        # Add detailed report if available (without base64 image to save space)
        if "detailed_report" in response:
            report_copy = response["detailed_report"].copy()
            # Remove base64 image from Firestore to save space
            report_copy.pop('confusion_matrix_image', None)
            doc["detailed_report"] = convert_to_native_types(report_copy)
        
        doc_ref = db.collection("predictions").add(convert_to_native_types(doc))
        logger.info(f"Prediction logged: {doc_ref[1].id}")
        
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")


def check_model_health() -> Dict[str, Any]:
    """Check if model is available and working."""
    try:
        health = {
            "status": "unknown",
            "checks": {},
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
        
        # Check files exist
        health['checks']['model_file_exists'] = MODEL_LATEST.exists()
        health['checks']['preprocessor_exists'] = PREPROCESSOR_LATEST.exists()
        health['checks']['training_history_exists'] = TRAINING_HISTORY.exists()
        
        # Try loading
        try:
            model, preprocessor, metadata = load_model_and_preprocessor()
            health['checks']['model_loadable'] = model is not None
            health['checks']['preprocessor_loadable'] = preprocessor is not None
            
            if model and preprocessor:
                health['model_info'] = metadata
                
                # Add training metrics if available
                if 'training_metrics' in metadata:
                    health['performance'] = {
                        'accuracy': metadata['training_metrics'].get('test_accuracy', 0) * 100,
                        'precision': metadata['training_metrics'].get('precision', 0) * 100,
                        'recall': metadata['training_metrics'].get('recall', 0) * 100,
                        'f1_score': metadata['training_metrics'].get('f1_score', 0) * 100,
                    }
        except Exception as e:
            health['checks']['model_loadable'] = False
            health['checks']['load_error'] = str(e)
        
        # Overall status
        if health['checks'].get('model_loadable') and health['checks'].get('preprocessor_loadable'):
            health['status'] = "healthy"
            health['message'] = "Model is ready for predictions"
        else:
            health['status'] = "unhealthy"
            health['message'] = "Model not available"
        
        return health
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }


# Export public API
__all__ = [
    'run_prediction',
    'check_model_health',
]