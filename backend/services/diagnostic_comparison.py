#!/usr/bin/env python3
"""
Diagnostic script to compare test_prediction.py vs prediction_service.py processing
Run with: python backend/services/diagnostic_comparison.py
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import
from backend.services.training_service import MODEL_PATH, PREPROCESSOR_PATH

# Import normalize_input and other functions manually to avoid circular imports
MODELS_DIR = Path("models")
MODEL_LATEST = MODELS_DIR / "burnout_latest.pkl"
PREPROCESSOR_LATEST = MODELS_DIR / "preprocessor_latest.pkl"

# Likert mapping
LIKERT_MAP = {
    "strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5,
    "strongly_disagree": 1, "strongly_agree": 5,
    "argee": 4, "agre": 4, "neural": 3, "nuetral": 3,
    "disargee": 2, "disagre": 2, "strongly argee": 5, "strongly disagre": 1,
    "never": 1, "rarely": 2, "sometimes": 3, "often": 4, "always": 5,
    "very low": 1, "low": 2, "medium": 3, "moderate": 3, "high": 4, "very high": 5,
    "no": 1, "yes": 5,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


def normalize_input_local(payload):
    """Local copy of normalize_input to avoid import issues."""
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


def load_preprocessor():
    """Load the preprocessor."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    return joblib.load(PREPROCESSOR_PATH)


def test_script_method(input_data):
    """Process data using test_prediction.py method."""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    clf = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Convert to DataFrame if dict
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Ensure correct column order
    feature_names = preprocessor['feature_names']
    df = df[feature_names]
    
    # Convert everything to numeric first
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
    
    # Convert to numpy array
    X = df.to_numpy()
    
    # Impute and scale
    X_imputed = preprocessor['imputer'].transform(X)
    X_scaled = preprocessor['scaler'].transform(X_imputed)
    
    # Predict
    prediction = clf.predict(X_scaled)[0]
    probabilities = clf.predict_proba(X_scaled)[0]
    classes = clf.classes_
    
    return {
        'method': 'test_script',
        'prediction': prediction,
        'probabilities': {str(cls): float(prob * 100) for cls, prob in zip(classes, probabilities)},
        'confidence': float(max(probabilities) * 100),
        'df_before_numpy': df.copy(),
        'X_numpy': X.copy(),
        'X_imputed': X_imputed.copy(),
        'X_scaled': X_scaled.copy()
    }


def api_method(input_data):
    """Process data using prediction_service.py method."""
    model, preprocessor, metadata = load_model_and_preprocessor()
    
    # Normalize input
    normalized = normalize_input(input_data)
    
    # Predict
    result = predict_burnout_simple(normalized, model, preprocessor)
    
    # Get intermediate data for comparison
    feature_names = preprocessor['feature_names']
    df = pd.DataFrame([normalized])
    
    # Add missing features
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = np.nan
    
    df = df[feature_names]
    
    # Convert to numeric
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
    
    X = df.to_numpy()
    X_imputed = preprocessor['imputer'].transform(X)
    X_scaled = preprocessor['scaler'].transform(X_imputed)
    
    return {
        'method': 'api',
        'prediction': result['prediction'],
        'probabilities': result['probabilities'],
        'confidence': result['confidence'],
        'normalized_input': normalized,
        'df_before_numpy': df.copy(),
        'X_numpy': X.copy(),
        'X_imputed': X_imputed.copy(),
        'X_scaled': X_scaled.copy()
    }


def compare_dataframes(df1, df2, name1="Method 1", name2="Method 2"):
    """Compare two DataFrames column by column."""
    print(f"\n{'='*80}")
    print(f"DATAFRAME COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}")
    
    differences = []
    
    for col in df1.columns:
        val1 = df1[col].iloc[0]
        val2 = df2[col].iloc[0]
        
        # Check if values are different (accounting for NaN)
        if pd.isna(val1) and pd.isna(val2):
            continue
        elif pd.isna(val1) or pd.isna(val2):
            differences.append((col, val1, val2))
        elif abs(float(val1) - float(val2)) > 0.0001:
            differences.append((col, val1, val2))
    
    if differences:
        print(f"\n❌ Found {len(differences)} differences:")
        print(f"\n{'Feature':<35} {name1:>15} {name2:>15} {'Diff':>15}")
        print("-" * 80)
        for col, val1, val2 in differences[:20]:  # Show first 20
            diff = "NaN mismatch" if (pd.isna(val1) or pd.isna(val2)) else f"{float(val1) - float(val2):.6f}"
            print(f"{col:<35} {str(val1):>15} {str(val2):>15} {diff:>15}")
        
        if len(differences) > 20:
            print(f"\n... and {len(differences) - 20} more differences")
    else:
        print("\n✅ DataFrames are IDENTICAL")
    
    return len(differences)


def compare_arrays(arr1, arr2, name="Array"):
    """Compare two numpy arrays."""
    print(f"\n{'='*80}")
    print(f"ARRAY COMPARISON: {name}")
    print(f"{'='*80}")
    
    if arr1.shape != arr2.shape:
        print(f"❌ Shape mismatch: {arr1.shape} vs {arr2.shape}")
        return False
    
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_different = np.sum(diff > 0.0001)
    
    print(f"Shape: {arr1.shape}")
    print(f"Max difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")
    print(f"Number of elements with diff > 0.0001: {num_different}")
    
    if max_diff > 0.0001:
        print("\n❌ Arrays are DIFFERENT")
        # Show first few differences
        diff_indices = np.where(diff > 0.0001)[1][:10]
        print(f"\nFirst 10 different indices:")
        for idx in diff_indices:
            print(f"  Index {idx}: {arr1[0, idx]:.6f} vs {arr2[0, idx]:.6f} (diff: {diff[0, idx]:.6f})")
        return False
    else:
        print("\n✅ Arrays are IDENTICAL (within tolerance)")
        return True


def run_diagnostic():
    """Run full diagnostic comparison."""
    print("\n" + "🔬"*40)
    print("DIAGNOSTIC: Test Script vs API Method Comparison")
    print("🔬"*40)
    
    # Test data from your log
    test_data = {
        "sleep_less_6_hours": "3",
        "difficult_fall_asleep": "1",
        "wake_up_tired": "5",
        "frequent_headaches_fatigue": "5",
        "skip_meals_stress": "2",
        "physically_exhausted": "1",
        "multitask_deadlines": "5",
        "study_under_pressure": "3",
        "workload_unmanageable": "4",
        "rarely_free_time": "5",
        "struggle_organize": "5",
        "use_planner": "4",
        "sacrifice_sleep": "5",
        "workload_heavier_peers": "1",
        "emotionally_drained": "4",
        "hard_feel_excited": "4",
        "feel_helpless": "2",
        "feel_burned_out": "1",
        "feel_giving_up": "4",
        "irritated_easily": "2",
        "sense_dread": "2",
        "proud_achievements": "2",
        "not_accomplishing": "4",
        "confident_handling": "4",
        "question_efforts": "5",
        "feel_competent": "2",
        "underperforming_peers": "5",
        "struggle_balance": "5",
        "waste_time": "1",
        "hard_maintain_routine": "3",
        "finish_deadline": "5",
        "feel_control": "5",
        "supported_family": "1",
        "feel_isolated": "4",
        "hesitate_ask_help": "5",
        "someone_talk": "4",
        "feel_disconnected": "3",
        "financial_difficulties": "1",
        "pressure_support_family": "1",
        "conflicts_home": "3",
        "noisy_home": "1",
        "emotionally_unsupported": "5",
        "miss_schoolwork": "5",
        "family_not_understand": "3",
        "personal_life_affects": "1",
        "traveling_affects_energy": "5",
        "commutes_stress": "2",
        "lose_motivation_commute": "5",
        "hard_focus": "4",
        "learning_setup_stress": "2",
        "feel_more_isolated": "5",
        "prefer_modality": "5",
        "mode_affects_performance": "4"
    }
    
    print("\n📋 Input Data:")
    print(f"   Total features: {len(test_data)}")
    
    # Run both methods
    print("\n" + "="*80)
    print("STEP 1: Running Test Script Method...")
    print("="*80)
    result_test = test_script_method(test_data)
    print(f"✅ Prediction: {result_test['prediction']}")
    print(f"✅ Confidence: {result_test['confidence']:.2f}%")
    print(f"✅ Probabilities:")
    for cls, prob in sorted(result_test['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"     {cls}: {prob:.2f}%")
    
    print("\n" + "="*80)
    print("STEP 2: Running API Method...")
    print("="*80)
    result_api = api_method(test_data)
    print(f"✅ Prediction: {result_api['prediction']}")
    print(f"✅ Confidence: {result_api['confidence']:.2f}%")
    print(f"✅ Probabilities:")
    for cls, prob in sorted(result_api['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"     {cls}: {prob:.2f}%")
    
    # Compare results
    print("\n" + "🔍"*40)
    print("COMPARISON RESULTS")
    print("🔍"*40)
    
    print(f"\n{'Metric':<30} {'Test Script':>20} {'API':>20} {'Match':>10}")
    print("-" * 80)
    print(f"{'Prediction':<30} {result_test['prediction']:>20} {result_api['prediction']:>20} {'✅' if result_test['prediction'] == result_api['prediction'] else '❌':>10}")
    print(f"{'Confidence':<30} {result_test['confidence']:>20.2f} {result_api['confidence']:>20.2f} {'✅' if abs(result_test['confidence'] - result_api['confidence']) < 0.01 else '❌':>10}")
    
    # Compare intermediate steps
    print("\n" + "="*80)
    print("STEP 3: Comparing Intermediate Processing Steps")
    print("="*80)
    
    # Compare DataFrames before numpy conversion
    diff_count = compare_dataframes(
        result_test['df_before_numpy'],
        result_api['df_before_numpy'],
        "Test Script DF",
        "API DF"
    )
    
    # Compare numpy arrays
    compare_arrays(result_test['X_numpy'], result_api['X_numpy'], "Raw Numpy Array (after DF conversion)")
    compare_arrays(result_test['X_imputed'], result_api['X_imputed'], "Imputed Array")
    compare_arrays(result_test['X_scaled'], result_api['X_scaled'], "Scaled Array (final input to model)")
    
    # Summary
    print("\n" + "📊"*40)
    print("DIAGNOSTIC SUMMARY")
    print("📊"*40)
    
    if result_test['prediction'] == result_api['prediction'] and abs(result_test['confidence'] - result_api['confidence']) < 0.01:
        print("\n✅ RESULTS MATCH - Both methods produce identical predictions")
    else:
        print("\n❌ RESULTS DIFFER - Methods produce different predictions")
        print("\n🔍 Root Cause Analysis:")
        if diff_count > 0:
            print(f"   • DataFrame preprocessing differs ({diff_count} features affected)")
            print(f"   • This causes different input to the model")
            print(f"   • Leading to different predictions")
            print("\n💡 Action Required:")
            print("   • Check the normalize_input() function in prediction_service.py")
            print("   • Compare with test_prediction.py data handling")
            print("   • Ensure both use identical preprocessing steps")
        else:
            print("   • DataFrames are identical but predictions differ")
            print("   • This suggests a model loading or prediction issue")
            print("   • Check if both are using the same model file")


if __name__ == "__main__":
    try:
        run_diagnostic()
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()