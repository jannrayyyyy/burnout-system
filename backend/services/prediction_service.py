# backend/services/prediction_service.py
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer

from backend.services.firebase_service import db

logger = logging.getLogger(__name__)

# Model locations (aligned with training_service)
MODELS_DIR = Path("models")
MODEL_LATEST = MODELS_DIR / "burnout_latest.pkl"
PREPROCESSOR_LATEST = MODELS_DIR / "preprocessor_latest.pkl"

# Enhanced Likert mapping (aligned with training - EXACT MATCH)
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
    return obj


# -------------------------
# Public API
# -------------------------
def load_model(path: Path = MODEL_LATEST) -> BaseEstimator:
    """
    Load trained model artifact.
    
    Args:
        path: Path to model file (default: burnout_latest.pkl)
        
    Returns:
        Trained sklearn estimator
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Please train a model first using the /admin/train endpoint."
        )
    
    try:
        model = joblib.load(path)
        logger.info(f"✅ Loaded model from {path}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise Exception(f"Model loading failed: {str(e)}")


def load_preprocessor(path: Path = PREPROCESSOR_LATEST) -> Optional[Dict[str, Any]]:
    """
    Load preprocessing artifacts (encoders, scaler, imputer).
    
    Args:
        path: Path to preprocessor file
        
    Returns:
        Dictionary containing preprocessing components or None if not found
    """
    if not path.exists():
        logger.warning(f"⚠️ Preprocessor not found at {path}. Using fallback preprocessing.")
        return None
    
    try:
        preprocessor = joblib.load(path)
        logger.info(f"✅ Loaded preprocessor from {path}")
        return preprocessor
    except Exception as e:
        logger.warning(f"⚠️ Failed to load preprocessor: {e}. Using fallback.")
        return None


def run_prediction(
    payload: Dict[str, Any],
    model: Optional[BaseEstimator] = None,
    preprocessor: Optional[Dict[str, Any]] = None,
    log_to_firestore: bool = True,
    user_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive burnout prediction with detailed explanations.
    
    This function:
    1. Normalizes and validates input data
    2. Applies the same preprocessing used during training
    3. Generates predictions with confidence scores
    4. Provides interpretable explanations
    5. Logs survey and prediction separately to Firestore
    
    Args:
        payload: Raw survey responses (dict with question keys and answers)
        model: Pre-loaded model (if None, loads from MODEL_LATEST)
        preprocessor: Pre-loaded preprocessor (if None, loads from PREPROCESSOR_LATEST)
        log_to_firestore: Whether to log prediction to database
        user_context: Optional metadata (user_id, session_id, etc.)
        
    Returns:
        Dictionary containing:
        - burnout_level: Predicted class (Low/Moderate/High)
        - probability: Confidence score (0-1)
        - all_probabilities: Scores for all classes
        - confidence_level: Human-readable confidence description
        - risk_category: Clinical interpretation
        - explanation: Feature contributions and interpretive summary
        - recommendations: Personalized suggestions based on prediction
        - model_info: Version and metadata
        - survey_id: Firestore document ID of the logged survey
        - prediction_id: Firestore document ID of the logged prediction
        
    Raises:
        ValueError: If payload is invalid
        RuntimeError: If prediction fails
    """
    try:
        # ========== PHASE 1: MODEL & PREPROCESSOR LOADING ==========
        if model is None:
            model = load_model()
        
        if preprocessor is None:
            preprocessor = load_preprocessor()
        
        if model is None:
            raise RuntimeError("Model could not be loaded. Train a model first.")
        
        logger.info("🔮 Starting prediction pipeline...")
        
        # ========== PHASE 2: INPUT VALIDATION & NORMALIZATION ==========
        if not isinstance(payload, dict) or len(payload) == 0:
            raise ValueError("Payload must be a non-empty dictionary")
        
        logger.info(f"📥 Received {len(payload)} input features")
        
        # Normalize and map Likert responses (EXACT MATCH to training)
        df_row = _normalize_and_map_payload(payload)
        logger.info(f"✅ Normalized input to DataFrame: {df_row.shape}")
        
        # ========== PHASE 3: FEATURE ALIGNMENT ==========
        if preprocessor and 'feature_names' in preprocessor:
            model_features = preprocessor['feature_names']
            logger.info(f"📊 Using preprocessor feature names: {len(model_features)} features")
        else:
            model_features = _get_model_feature_names(model)
            logger.info(f"📊 Extracted feature names from model: {len(model_features)} features")
        
        if not model_features:
            raise RuntimeError(
                "Could not extract feature names from model. "
                "Model may be incompatible or corrupted."
            )
        
        df_aligned = _align_features(df_row, model_features)
        
        # ========== PHASE 4: PREPROCESSING (EXACT MATCH to training) ==========
        if preprocessor:
            df_prepared = _apply_saved_preprocessing(df_aligned, preprocessor)
            logger.info("✅ Applied saved preprocessing pipeline (RobustScaler)")
        else:
            df_prepared = _fallback_preprocessing(df_aligned)
            logger.info("⚠️ Using fallback preprocessing (no saved preprocessor)")
        
        # ========== PHASE 5: PREDICTION ==========
        try:
            proba = _predict_proba_safe(model, df_prepared)
            logger.info(f"✅ Prediction completed: {proba}")
        except Exception as e:
            logger.exception("❌ Prediction failed")
            raise RuntimeError(f"Prediction execution failed: {str(e)}")
        
        classes = _get_model_classes(model)
        best_idx = int(np.argmax(proba))
        best_class = str(classes[best_idx])
        best_prob = float(proba[best_idx])
        
        logger.info(f"🎯 Predicted: {best_class} (confidence: {best_prob:.2%})")
        
        # ========== PHASE 6: EXPLANATION & INTERPRETATION ==========
        explanation = _explain_prediction(
            model=model,
            df=df_prepared,
            feature_names=model_features,
            predicted_class=best_class,
            top_k=8
        )
        
        confidence_level = _interpret_confidence(best_prob)
        risk_category = _interpret_risk(best_class, best_prob)
        recommendations = _generate_recommendations(best_class, best_prob, explanation)
        
        # ========== PHASE 7: MODEL METADATA ==========
        model_info = _extract_model_info(model)
        
        # ========== PHASE 8: FIRESTORE LOGGING (SEPARATE SURVEY & PREDICTION) ==========
        survey_id = None
        prediction_id = None
        
        if log_to_firestore:
            try:
                # Step 1: Log prediction FIRST (to get prediction_id)
                prediction_id = _log_prediction_to_firestore(
                    prediction=best_class,
                    confidence=best_prob,
                    all_probabilities={
                        str(c): float(p) 
                        for c, p in zip(map(str, classes), proba.tolist())
                    },
                    model_info=model_info,
                    explanation=explanation,
                    recommendations=recommendations,
                    confidence_level=confidence_level,
                    risk_category=risk_category,
                    user_context=user_context,
                )
                logger.info(f"✅ Prediction logged to Firestore with ID: {prediction_id}")
                
                # Step 2: Log survey with predictionId reference
                survey_id = _log_survey_to_firestore(
                    payload=payload,
                    cleaned_row=df_prepared.iloc[0].to_dict(),
                    prediction_id=prediction_id,
                    user_context=user_context,
                )
                logger.info(f"✅ Survey logged to Firestore with ID: {survey_id}")
                
            except Exception as e:
                logger.warning(f"⚠️ Firestore logging failed (non-fatal): {e}")
        
        # ========== PHASE 9: BUILD RESPONSE ==========
        response = {
            "burnout_level": best_class,
            "probability": round(best_prob, 4),
            "all_probabilities": {
                str(c): round(float(p), 4) 
                for c, p in zip(map(str, classes), proba.tolist())
            },
            "confidence_level": confidence_level,
            "risk_category": risk_category,
            "explanation": explanation,
            "recommendations": recommendations,
            "model_info": {
                "version": model_info.get("version"),
                "records_used": model_info.get("records_used"),
                "created_at": model_info.get("created_at"),
                "accuracy": model_info.get("accuracy"),
            },
            "prediction_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "features_used": len(model_features),
                "preprocessing_method": "saved_pipeline" if preprocessor else "fallback",
                "survey_id": survey_id,
                "prediction_id": prediction_id,
            }
        }
        
        logger.info("🎉 Prediction pipeline completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction pipeline failed: {str(e)}", exc_info=True)
        raise


# -------------------------
# Helper Functions
# -------------------------
def _normalize_and_map_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalize input payload to match training data format.
    EXACT MATCH to training_service clean_and_normalize_data + advanced_likert_mapping
    
    - Converts keys to lowercase with underscores
    - Maps Likert scale responses to numeric values (1-5)
    - Handles common typos and variations
    - Coerces numeric strings to floats
    
    Args:
        payload: Raw input dictionary
        
    Returns:
        Single-row DataFrame with normalized features
    """
    if not isinstance(payload, dict):
        raise ValueError("Prediction payload must be a JSON object/dict")
    
    clean = {}
    empty_values = ["", " ", "nan", "NaN", "NA", "N/A", "null", "None", "#N/A", "?", "--"]
    
    for k, v in payload.items():
        if not isinstance(k, str):
            continue
        
        # Normalize key (EXACT MATCH to training)
        key = k.strip().lower()
        key = key.replace(" ", "_").replace("-", "_")
        key = key.replace("(", "").replace(")", "")
        key = key.replace("[", "").replace("]", "")
        
        # Handle empty values
        if v in empty_values:
            clean[key] = np.nan
            continue
        
        # Handle Likert string responses
        if isinstance(v, str):
            vs = v.strip()
            
            # Try Likert mapping first (case-insensitive)
            mapped = LIKERT_MAP.get(vs.lower())
            if mapped is not None:
                clean[key] = mapped
                continue
            
            # Try numeric conversion
            try:
                clean[key] = float(vs.replace(",", ""))
                continue
            except (ValueError, AttributeError):
                pass
            
            # Keep as string for later encoding
            clean[key] = vs
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            clean[key] = np.nan
        else:
            clean[key] = v
    
    if not clean:
        raise ValueError("No valid features found in payload after normalization")
    
    df = pd.DataFrame([clean])
    
    # Replace empty values with NaN
    df.replace(empty_values, np.nan, inplace=True)
    
    logger.debug(f"Normalized {len(clean)} features from payload")
    
    return df


def _get_model_feature_names(model: BaseEstimator) -> List[str]:
    """
    Extract expected feature names from trained model.
    
    Supports:
    - RandomForestClassifier with feature_names_in_
    - Pipeline with preprocessor
    - Models with n_features_in_
    """
    try:
        # Try feature_names_in_ (sklearn >= 1.0)
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        
        # Try n_features_in_ as fallback
        if hasattr(model, "n_features_in_"):
            n = model.n_features_in_
            return [f"feature_{i}" for i in range(n)]
        
        logger.warning("Could not extract feature names from model")
        return []
        
    except Exception as e:
        logger.error(f"Error extracting feature names: {e}")
        return []


def _align_features(df: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
    """
    Align input DataFrame to match model's expected features.
    
    - Adds missing columns as NaN
    - Removes extra columns
    - Preserves feature order
    """
    df = df.copy()
    
    # Add missing features
    for feat in model_features:
        if feat not in df.columns:
            df[feat] = np.nan
    
    # Select and order features
    df_aligned = df[model_features].copy()
    
    missing_count = df_aligned.isna().sum().sum()
    if missing_count > 0:
        logger.debug(f"⚠️ {missing_count} missing values will be imputed")
    
    return df_aligned


def _apply_saved_preprocessing(df: pd.DataFrame, preprocessor: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply saved preprocessing pipeline from training.
    EXACT MATCH to training_service preprocessing steps.
    
    Uses the exact same transformations:
    - Label encoding for categorical features
    - Median imputation for missing values
    - RobustScaler (not StandardScaler - matching training!)
    """
    df = df.copy()
    
    # Extract preprocessing components
    label_encoders = preprocessor.get('label_encoders', {})
    imputer = preprocessor.get('imputer')
    scaler = preprocessor.get('scaler')  # RobustScaler from training
    cat_cols = preprocessor.get('categorical_columns', [])
    
    # Step 1: Apply label encoding for categorical columns
    for col in cat_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            try:
                # Handle unseen categories
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if pd.notna(x) and str(x) in le.classes_ else -1
                )
            except Exception as e:
                logger.warning(f"Label encoding failed for {col}: {e}")
                df[col] = -1
    
    # Step 2: Apply imputation (median strategy, matching training)
    if imputer is not None:
        try:
            df_imputed = pd.DataFrame(
                imputer.transform(df),
                columns=df.columns,
                index=df.index
            )
        except Exception as e:
            logger.warning(f"Imputation failed: {e}. Using median fallback.")
            df_imputed = df.fillna(df.median()).fillna(0)
    else:
        df_imputed = df.fillna(df.median()).fillna(0)
    
    # Step 3: Apply scaling (RobustScaler, matching training)
    if scaler is not None:
        try:
            df_scaled = pd.DataFrame(
                scaler.transform(df_imputed),
                columns=df_imputed.columns,
                index=df_imputed.index
            )
        except Exception as e:
            logger.warning(f"Scaling failed: {e}. Using data as-is.")
            df_scaled = df_imputed
    else:
        df_scaled = df_imputed
    
    return df_scaled


def _fallback_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback preprocessing when saved preprocessor is unavailable.
    
    Applies basic transformations:
    - Numeric coercion
    - Median imputation
    - Simple standardization
    """
    df = df.copy()
    
    # Coerce to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Impute with median
    df = df.fillna(df.median()).fillna(0)
    
    # Simple standardization
    means = df.mean()
    stds = df.std().replace(0, 1)  # Avoid division by zero
    df_scaled = (df - means) / stds
    
    return df_scaled


def _predict_proba_safe(model: BaseEstimator, df: pd.DataFrame) -> np.ndarray:
    """
    Safely extract probability predictions from model.
    
    Returns:
        1D array of probabilities for single sample
    """
    if not hasattr(model, 'predict_proba'):
        raise RuntimeError("Model does not support probability predictions")
    
    try:
        proba = model.predict_proba(df)
        return proba[0]  # Return first (and only) row
    except Exception as e:
        logger.error(f"predict_proba failed: {e}")
        raise


def _get_model_classes(model: BaseEstimator) -> np.ndarray:
    """Extract class labels from model."""
    if hasattr(model, 'classes_'):
        return model.classes_
    return np.array(['Low', 'Moderate', 'High'])  # Fallback


def _explain_prediction(
    model: BaseEstimator,
    df: pd.DataFrame,
    feature_names: List[str],
    predicted_class: str,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Generate comprehensive explanation of prediction.
    
    Uses feature importances to identify:
    - Most influential factors
    - Their contribution to the prediction
    - Human-readable interpretations
    
    Returns:
        Dictionary with top_features, summary, and clinical_notes
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            # Fallback explanation when feature importances aren't available
            fallback_summary = f"Your burnout level is predicted as **{predicted_class}**. "
            if predicted_class == "High":
                fallback_summary += "This suggests you're experiencing significant symptoms of burnout including emotional exhaustion, cynicism, and reduced sense of accomplishment. Multiple stressors are likely contributing to your current state."
            elif predicted_class == "Moderate":
                fallback_summary += "This indicates you're experiencing warning signs of burnout. You may notice increased fatigue, decreased motivation, and difficulty managing stress. Early intervention can prevent progression to severe burnout."
            else:
                fallback_summary += "This suggests you're managing stress relatively well and maintaining healthy coping mechanisms. Continue your current wellness practices to maintain this positive state."
            
            fallback_notes = _generate_clinical_notes(predicted_class, [])
            
            return {
                "summary": fallback_summary,
                "top_features": [],
                "clinical_notes": fallback_notes,
                "total_features_analyzed": len(feature_names)
            }
        
        importances = np.array(model.feature_importances_)
        
        if len(importances) != len(feature_names):
            logger.warning(f"Feature count mismatch: {len(importances)} vs {len(feature_names)}")
            n = min(len(importances), len(feature_names))
            importances = importances[:n]
            feature_names = feature_names[:n]
        
        # Calculate contribution scores
        row = df.iloc[0]
        contributions = []
        
        for fname, importance in zip(feature_names, importances):
            value = row.get(fname, 0)
            
            try:
                # Contribution = importance × normalized value
                contrib_score = abs(importance) * abs(float(value)) / (abs(float(value)) + 1.0)
            except:
                contrib_score = abs(importance)
            
            contributions.append({
                "feature": fname,
                "importance": float(importance),
                "value": float(value),
                "contribution": float(contrib_score)
            })
        
        # Normalize contributions to percentages
        total_contrib = sum(c['contribution'] for c in contributions) or 1.0
        for c in contributions:
            c['contribution_pct'] = round((c['contribution'] / total_contrib) * 100, 1)
        
        # Sort and select top features
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        top_features = contributions[:top_k]
        
        # Generate human-readable summary
        summary_parts = []
        for feat in top_features[:5]:  # Top 5 for summary
            label = _humanize_feature_name(feat['feature'])
            pct = feat['contribution_pct']
            value = feat.get('value', 0)
            
            # Determine impact level
            if pct >= 15:
                level = "major"
                impact = "significantly driving"
            elif pct >= 8:
                level = "significant"
                impact = "notably contributing to"
            elif pct >= 4:
                level = "moderate"
                impact = "moderately influencing"
            else:
                level = "minor"
                impact = "slightly affecting"
            
            # Create detailed explanation with context
            explanation = f"**{label}** is a {level} factor ({pct}% contribution) - {impact} your burnout level"
            
            # Add interpretation based on the value
            if value > 0.5:  # Higher values indicate more concerning responses
                explanation += ". Your responses in this area suggest elevated concern"
            elif value < -0.5:
                explanation += ". Your responses here show positive indicators"
            
            summary_parts.append(explanation)
        
        # Create comprehensive summary text
        if summary_parts:
            summary = "**Analysis of Key Contributing Factors:**\n\n"
            summary += ". ".join(summary_parts) + ".\n\n"
            summary += f"The model analyzed {len(feature_names)} different factors from your survey responses. "
            summary += f"The prediction is based on patterns identified across multiple dimensions of burnout including "
            summary += f"emotional exhaustion, cynicism/depersonalization, and personal accomplishment."
        else:
            summary = f"Your burnout level is predicted as **{predicted_class}** based on comprehensive analysis of {len(feature_names)} factors."
        # Clinical interpretation
        clinical_notes = _generate_clinical_notes(predicted_class, top_features)
        
        return {
            "top_features": top_features,
            "summary": summary,
            "clinical_notes": clinical_notes,
            "total_features_analyzed": len(feature_names)
        }
        
    except Exception as e:
        logger.exception(f"Explanation generation failed: {e}")
        return {
            "summary": f"Explanation unavailable: {str(e)}",
            "top_features": [],
            "clinical_notes": ""
        }


def _humanize_feature_name(feature: str) -> str:
    """Convert feature name to readable format."""
    # Remove common prefixes/suffixes
    name = feature.replace("_", " ")
    name = name.replace("how often do you", "")
    name = name.replace("rate your", "")
    
    # Capitalize
    name = " ".join(word.capitalize() for word in name.split())
    
    return name.strip()


def _generate_clinical_notes(predicted_class: str, top_features: List[Dict]) -> str:
    """Generate comprehensive clinical interpretation based on top features and burnout level."""
    
    # Base clinical assessment by burnout level
    if predicted_class == "High":
        base = "⚠️ **CLINICAL ASSESSMENT: High Burnout Risk Detected**\n\n"
        base += "**Severity:** Your assessment indicates severe burnout characterized by significant emotional exhaustion, "
        base += "depersonalization/cynicism, and markedly reduced personal accomplishment. This level of burnout is associated with "
        base += "serious health consequences including depression, anxiety disorders, cardiovascular issues, and immune system compromise.\n\n"
        
    elif predicted_class == "Moderate":
        base = "⚡ **CLINICAL ASSESSMENT: Moderate Burnout Risk**\n\n"
        base += "**Severity:** Your results indicate moderate burnout with noticeable symptoms affecting daily functioning. "
        base += "You're experiencing warning signs including increased fatigue, emotional volatility, difficulty concentrating, "
        base += "and declining motivation. Without intervention, there's significant risk of progression to severe burnout.\n\n"
        
    else:
        base = "✅ **CLINICAL ASSESSMENT: Low Burnout Risk**\n\n"
        base += "**Severity:** Your assessment indicates minimal burnout symptoms. You appear to be managing stressors effectively "
        base += "and maintaining adequate coping mechanisms. However, remain vigilant for early warning signs, especially during "
        base += "periods of increased demand or life transitions.\n\n"
    
    # Analyze contributing factor categories
    if top_features and len(top_features) > 0:
        base += "**Primary Contributing Factors:**\n"
        
        categories_found = {
            'sleep': False,
            'workload': False,
            'emotional': False,
            'accomplishment': False,
            'social': False,
            'physical': False
        }
        
        category_details = []
        
        for feat in top_features[:5]:
            fname = feat['feature'].lower()
            contrib = feat.get('contribution_pct', 0)
            
            if any(word in fname for word in ['sleep', 'rest', 'fatigue', 'tired', 'exhausted']) and not categories_found['sleep']:
                categories_found['sleep'] = True
                category_details.append(
                    f"• **Sleep Disturbance ({contrib}%):** Disrupted sleep patterns or insufficient rest are significantly "
                    f"impacting your burnout level. Sleep deprivation creates a cascade of negative effects including impaired "
                    f"cognitive function, emotional dysregulation, and weakened stress resilience."
                )
                
            elif any(word in fname for word in ['work', 'academic', 'load', 'deadline', 'task', 'responsibility']) and not categories_found['workload']:
                categories_found['workload'] = True
                category_details.append(
                    f"• **Workload Pressure ({contrib}%):** Excessive demands, unrealistic deadlines, or poor work-life boundaries "
                    f"are major contributors. Chronic overload depletes psychological resources and creates sustained stress response activation."
                )
                
            elif any(word in fname for word in ['emotional', 'stress', 'anxiety', 'overwhelm', 'pressure', 'worry']) and not categories_found['emotional']:
                categories_found['emotional'] = True
                category_details.append(
                    f"• **Emotional Stress ({contrib}%):** High emotional demands, anxiety, or feeling overwhelmed are significantly "
                    f"affecting your wellbeing. Prolonged emotional strain without adequate recovery leads to emotional exhaustion, "
                    f"a core component of burnout."
                )
                
            elif any(word in fname for word in ['motivation', 'accomplish', 'achievement', 'competent', 'confidence', 'giving_up']) and not categories_found['accomplishment']:
                categories_found['accomplishment'] = True
                category_details.append(
                    f"• **Reduced Personal Accomplishment ({contrib}%):** Decreased sense of achievement, competence, or motivation "
                    f"is contributing to burnout. This often manifests as feelings of ineffectiveness, lack of purpose, or questioning "
                    f"your abilities despite objective evidence of competence."
                )
                
            elif any(word in fname for word in ['social', 'support', 'relationship', 'connection', 'isolated']) and not categories_found['social']:
                categories_found['social'] = True
                category_details.append(
                    f"• **Social Support Deficit ({contrib}%):** Limited social connections, lack of supportive relationships, or "
                    f"social isolation are contributing factors. Strong social networks are protective against burnout and essential "
                    f"for emotional resilience."
                )
                
            elif any(word in fname for word in ['physical', 'health', 'exercise', 'activity']) and not categories_found['physical']:
                categories_found['physical'] = True
                category_details.append(
                    f"• **Physical Health Factors ({contrib}%):** Physical wellbeing indicators suggest concerns with exercise, "
                    f"general health, or physical activity levels. Physical and mental health are deeply interconnected in burnout development."
                )
        
        if category_details:
            base += "\n".join(category_details)
        else:
            base += "• Multiple psychosocial and environmental factors are contributing to your current burnout level. "
            base += "A comprehensive assessment of work demands, personal resources, and coping strategies is recommended."
        
        base += "\n\n"
    
    else:
        # When no top features available
        base += "**Primary Contributing Factors:**\n"
        base += "• Detailed factor analysis is limited, but burnout typically develops from a combination of chronic stressors "
        base += "including excessive demands, insufficient resources, lack of control, inadequate social support, and misalignment "
        base += "between personal values and work/academic environment.\n\n"
    
    # Add clinical recommendations section
    base += "**Clinical Recommendations:**\n"
    
    if predicted_class == "High":
        base += "• **Immediate professional evaluation recommended** - Consider scheduling with a mental health professional within 48-72 hours\n"
        base += "• Screen for comorbid conditions (depression, anxiety, substance use) as these commonly co-occur with severe burnout\n"
        base += "• Evaluate need for temporary work/academic accommodations or leave\n"
        base += "• Implement crisis management strategies immediately\n"
        base += "• Consider short-term interventions: cognitive-behavioral therapy (CBT), stress management training, or mindfulness-based stress reduction (MBSR)\n"
        
    elif predicted_class == "Moderate":
        base += "• **Preventive intervention recommended** - Consult with a counselor or therapist to develop coping strategies\n"
        base += "• Conduct comprehensive life-balance audit to identify modifiable stressors\n"
        base += "• Implement structured stress management program\n"
        base += "• Monitor symptoms weekly; if worsening, escalate to more intensive intervention\n"
        base += "• Consider group-based interventions: stress management workshops, peer support groups\n"
        
    else:
        base += "• **Continue current wellness practices** - Your current strategies are working effectively\n"
        base += "• Implement regular self-monitoring (monthly check-ins) to detect early warning signs\n"
        base += "• Maintain work-life boundaries and protective factors\n"
        base += "• Consider this assessment quarterly or when facing significant life changes\n"
        base += "• Share successful strategies with peers who may be struggling\n"
    
    base += "\n**Remember:** Burnout is a **treatable condition**, not a personal failure. Seeking help is a sign of strength and self-awareness."
    
    return base


def _interpret_confidence(probability: float) -> str:
    """Convert probability to human-readable confidence level."""
    if probability >= 0.85:
        return "Very High (85%+)"
    elif probability >= 0.70:
        return "High (70-85%)"
    elif probability >= 0.55:
        return "Moderate (55-70%)"
    else:
        return "Low (below 55%)"


def _interpret_risk(burnout_level: str, probability: float) -> str:
    """Generate clinical risk interpretation."""
    risk_matrix = {
        ("High", "high"): "Critical - Immediate intervention recommended",
        ("High", "moderate"): "Elevated - Professional assessment advised",
        ("Moderate", "high"): "Concerning - Monitor closely and implement coping strategies",
        ("Moderate", "moderate"): "Watchful - Preventive measures suggested",
        ("Low", "high"): "Stable - Continue current wellness practices",
        ("Low", "moderate"): "Healthy - Maintain balanced lifestyle",
    }
    
    conf = "high" if probability >= 0.70 else "moderate"
    key = (burnout_level, conf)
    
    return risk_matrix.get(key, "Assessment complete - Review recommendations")


def _generate_recommendations(
    burnout_level: str,
    probability: float,
    explanation: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate personalized recommendations based on prediction."""
    
    recommendations = []
    
    # Risk-based recommendations
    if burnout_level == "High":
        recommendations.append({
            "category": "Immediate Action",
            "priority": "urgent",
            "suggestion": "Consult with a mental health professional or counselor to develop a comprehensive wellness plan."
        })
        recommendations.append({
            "category": "Workload Management",
            "priority": "high",
            "suggestion": "Reassess current commitments and consider reducing workload or academic load temporarily."
        })
    elif burnout_level == "Moderate":
        recommendations.append({
            "category": "Preventive Care",
            "priority": "high",
            "suggestion": "Implement stress management techniques such as mindfulness, meditation, or regular exercise."
        })
    else:
        recommendations.append({
            "category": "Maintenance",
            "priority": "normal",
            "suggestion": "Continue current wellness practices and maintain work-life balance."
        })
    
    # Feature-specific recommendations
    top_features = explanation.get('top_features', [])
    for feat in top_features[:3]:
        fname = feat['feature'].lower()
        
        if 'sleep' in fname:
            recommendations.append({
                "category": "Sleep Hygiene",
                "priority": "high" if burnout_level != "Low" else "normal",
                "suggestion": "Establish consistent sleep schedule (7-9 hours), avoid screens before bed, create relaxing bedtime routine."
            })
        elif any(word in fname for word in ['work', 'academic', 'load']):
            recommendations.append({
                "category": "Time Management",
                "priority": "high",
                "suggestion": "Use prioritization techniques (Eisenhower Matrix), break tasks into smaller chunks, schedule regular breaks."
            })
        elif any(word in fname for word in ['emotional', 'stress']):
            recommendations.append({
                "category": "Emotional Wellbeing",
                "priority": "high",
                "suggestion": "Practice daily relaxation (deep breathing, progressive muscle relaxation), journal emotions, seek social support."
            })
        elif 'social' in fname:
            recommendations.append({
                "category": "Social Connection",
                "priority": "normal",
                "suggestion": "Maintain regular contact with friends and family, join support groups or social activities."
            })
    
    # Add general wellness recommendation
    recommendations.append({
        "category": "Holistic Wellness",
        "priority": "normal",
        "suggestion": "Maintain balanced diet, stay physically active (30 min/day), practice gratitude, limit caffeine and alcohol."
    })
    
    # Remove duplicates by category
    seen_categories = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec['category'] not in seen_categories:
            seen_categories.add(rec['category'])
            unique_recommendations.append(rec)
    
    return unique_recommendations[:6]  # Limit to 6 recommendations


def _extract_model_info(model: BaseEstimator) -> Dict[str, Any]:
    """Extract comprehensive model metadata from file system and Firestore."""
    info = {
        "version": None,
        "records_used": None,
        "created_at": None,
        "accuracy": None,
        "model_type": None,
        "features_count": None
    }
    
    try:
        # Get model type
        info["model_type"] = type(model).__name__
        
        # Get feature count
        if hasattr(model, "n_features_in_"):
            info["features_count"] = int(model.n_features_in_)
        
        # Get version from saved models count
        version_files = list(MODELS_DIR.glob("burnout_v*.pkl"))
        if version_files:
            # Extract version numbers and get the max
            versions = []
            for f in version_files:
                try:
                    v = int(f.stem.split('_v')[1])
                    versions.append(v)
                except:
                    pass
            if versions:
                info["version"] = max(versions)
        
        # Get file metadata
        if MODEL_LATEST.exists():
            stat = MODEL_LATEST.stat()
            info["created_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Try to load metadata from Firestore (most reliable source)
        if db:
            try:
                models_ref = db.collection("models").order_by(
                    "trained_at", direction="DESCENDING"
                ).limit(1).stream()
                
                for doc in models_ref:
                    data = doc.to_dict()
                    # Override with Firestore data if available (more reliable)
                    info["records_used"] = data.get("records_used") or data.get("sample_count")
                    info["accuracy"] = data.get("accuracy")
                    info["version"] = data.get("version", info["version"])
                    
                    # Get additional metadata
                    if "best_model" in data:
                        info["model_type"] = data.get("best_model")
                    
                    if "trained_at" in data:
                        trained_at = data.get("trained_at")
                        if hasattr(trained_at, 'isoformat'):
                            info["created_at"] = trained_at.isoformat()
                        elif isinstance(trained_at, str):
                            info["created_at"] = trained_at
                    
                    break
                    
            except Exception as e:
                logger.debug(f"Could not fetch model metadata from Firestore: {e}")
        
        # Set defaults if still None
        if info["version"] is None:
            info["version"] = 1
        if info["records_used"] is None:
            info["records_used"] = "Unknown"
        if info["accuracy"] is None:
            info["accuracy"] = "Not available"
        
    except Exception as e:
        logger.debug(f"Could not extract full model metadata: {e}")
    
    return info


def _log_prediction_to_firestore(
    prediction: str,
    confidence: float,
    all_probabilities: Dict[str, float],
    model_info: Dict[str, Any],
    explanation: Dict[str, Any],
    recommendations: List[Dict[str, str]],
    confidence_level: str,
    risk_category: str,
    user_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Log prediction results to Firestore 'predictions' collection.
    
    Stores:
    - Prediction result and confidence scores
    - Explanation and top contributing features
    - Recommendations
    - Model version and metadata
    - User context for auditing
    
    Returns:
        Firestore document ID of the logged prediction, or None if logging failed
    """
    if db is None:
        logger.debug("Firestore DB not available; skipping prediction logging")
        return None

    try:
        doc = {
            "prediction": prediction,
            "confidence": float(confidence),
            "all_probabilities": convert_to_native_types(all_probabilities),
            "confidence_level": confidence_level,
            "risk_category": risk_category,
            "explanation": {
                "summary": explanation.get("summary", ""),
                "clinical_notes": explanation.get("clinical_notes", ""),
                "top_features": [
                    {
                        "feature": f.get("feature", ""),
                        "contribution_pct": f.get("contribution_pct", 0),
                        "importance": f.get("importance", 0),
                        "value": f.get("value", 0)
                    } for f in explanation.get("top_features", [])
                ],
                "total_features_analyzed": explanation.get("total_features_analyzed", 0)
            },
            "recommendations": convert_to_native_types(recommendations),
            "model_info": {
                "version": model_info.get("version"),
                "created_at": model_info.get("created_at"),
                "accuracy": model_info.get("accuracy"),
                "records_used": model_info.get("records_used")
            },
            "user_context": convert_to_native_types(user_context or {}),
            "timestamp": datetime.utcnow(),
            "status": "completed"
        }

        # Store in Firestore under "predictions" collection
        doc_ref = db.collection("predictions").add(convert_to_native_types(doc))
        prediction_id = doc_ref[1].id
        
        logger.info(f"✅ Prediction successfully logged to Firestore with ID: {prediction_id}")
        return prediction_id

    except Exception as e:
        logger.warning(f"⚠️ Failed to log prediction to Firestore: {e}")
        return None


def _log_survey_to_firestore(
    payload: Dict[str, Any],
    cleaned_row: Dict[str, Any],
    prediction_id: Optional[str],
    user_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Log survey responses to Firestore 'surveys' collection.
    
    Stores:
    - Raw survey responses (payload)
    - Cleaned/processed feature values
    - Reference to prediction document (predictionId)
    - User context and metadata
    
    Returns:
        Firestore document ID of the logged survey, or None if logging failed
    """
    if db is None:
        logger.debug("Firestore DB not available; skipping survey logging")
        return None

    try:
        # Convert numpy types to native Python for Firestore
        cleaned_row_native = convert_to_native_types(cleaned_row)
        
        # Ensure user_context is properly structured
        context = user_context or {}
        if not isinstance(context, dict):
            context = {}

        doc = {
            "raw_responses": convert_to_native_types(payload),
            "processed_features": cleaned_row_native,
            "predictionId": prediction_id,  # Reference to the prediction document
            "user_context": convert_to_native_types(context),
            "timestamp": datetime.utcnow(),
            "status": "completed",
            "feature_count": len(payload),
            "has_prediction": prediction_id is not None
        }

        # Store in Firestore under "surveys" collection
        doc_ref = db.collection("surveys").add(convert_to_native_types(doc))
        survey_id = doc_ref[1].id
        
        logger.info(f"✅ Survey successfully logged to Firestore with ID: {survey_id}")
        return survey_id

    except Exception as e:
        logger.warning(f"⚠️ Failed to log survey to Firestore: {e}")
        return None