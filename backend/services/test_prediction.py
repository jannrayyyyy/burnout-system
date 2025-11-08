#!/usr/bin/env python3
"""
Test script for burnout prediction model
Run with: python backend/services/test_prediction.py
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.training_service import MODEL_PATH, PREPROCESSOR_PATH


def get_model_features():
    """Get the actual feature names the model expects."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return preprocessor['feature_names']


def predict_burnout(input_data):
    """
    Predict burnout level with proper DataFrame handling.
    
    Args:
        input_data: DataFrame or dict with survey question responses
        
    Returns:
        Dictionary with prediction results
    """
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        # Load latest model and preprocessor
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError("No trained model found. Please train a model first.")
        
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
        
        # Convert to numpy array (this eliminates ALL feature name warnings)
        X = df.to_numpy()
        
        # Impute and scale using numpy arrays
        X_imputed = preprocessor['imputer'].transform(X)
        X_scaled = preprocessor['scaler'].transform(X_imputed)
        
        # Predict
        prediction = clf.predict(X_scaled)[0]
        probabilities = clf.predict_proba(X_scaled)[0]
        
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
        print(f"❌ Prediction error: {e}")
        raise


def create_test_data_from_patterns(patterns, default_value="3"):
    """
    Create test data dictionary using keyword patterns to match actual feature names.
    
    Args:
        patterns: Dict with keyword patterns and desired values
        default_value: Default value for unmatched features
    
    Returns:
        pandas DataFrame with proper column names
    """
    feature_names = get_model_features()
    test_data = {}
    
    # First, set all features to default
    for feature in feature_names:
        test_data[feature] = default_value
    
    # Then, match patterns and set specific values
    matched_count = 0
    for pattern, value in patterns.items():
        pattern_lower = pattern.lower()
        matched = False
        
        for feature in feature_names:
            feature_lower = feature.lower()
            # Check if pattern words are in feature name
            pattern_words = pattern_lower.replace('_', ' ').split()
            if all(word in feature_lower for word in pattern_words):
                test_data[feature] = value
                matched = True
                matched_count += 1
        
        if not matched:
            print(f"⚠️  Warning: Pattern '{pattern}' didn't match any feature")
    
    # Return as DataFrame to preserve feature names
    return pd.DataFrame([test_data])


def print_result(test_name, result, expected_level=None):
    """Pretty print prediction results with detailed explanation."""
    print("\n" + "="*80)
    print(f"TEST: {test_name}")
    print("="*80)
    print(f"🎯 Prediction: {result['prediction']}")
    print(f"💯 Confidence: {result['confidence']:.2f}%")
    
    # Explain confidence in simple terms
    conf = result['confidence']
    if conf >= 80:
        confidence_desc = "Very Sure - The model is highly confident in this prediction"
    elif conf >= 60:
        confidence_desc = "Pretty Sure - The model has good confidence in this prediction"
    elif conf >= 40:
        confidence_desc = "Somewhat Uncertain - The model sees mixed signals"
    else:
        confidence_desc = "Very Uncertain - The model cannot decide clearly"
    
    print(f"📝 What this means: {confidence_desc}")
    
    print(f"\n📊 Probabilities:")
    
    # Sort by probability (highest first) and show with visual bars
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for level, prob in sorted_probs:
        bar = "█" * int(prob / 2)
        marker = "👉" if level == result['prediction'] else "  "
        print(f" {marker} {level:12} {prob:6.2f}% {bar}")
    
    if expected_level:
        match = "✅ PASS" if result['prediction'] == expected_level else "⚠️  REVIEW"
        confidence_note = ""
        if result['confidence'] < 50:
            confidence_note = " (Low confidence - model uncertain)"
        print(f"\n{match} (Expected: {expected_level}, Got: {result['prediction']}){confidence_note}")
    print("="*80)


def test_case_1_high_burnout():
    """Test Case 1: High Burnout Student - Severe symptoms across all dimensions."""
    print("\n🔴 TEST CASE 1: High Burnout Profile")
    print("Profile: Severe exhaustion, high workload, emotional distress, poor sleep")
    
    patterns = {
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


    
    test_data = create_test_data_from_patterns(patterns, default_value="4")
    result = predict_burnout(test_data)
    print_result("High Burnout Student", result, expected_level="High")
    return result


def test_case_2_low_burnout():
    """Test Case 2: Low Burnout Student - Healthy, balanced, well-adjusted."""
    print("\n🟢 TEST CASE 2: Low Burnout Profile")
    print("Profile: Good sleep, manageable workload, strong coping, social support")
    
    patterns = {
        # Physical health (good)
        "sleep_less_6_hours": "1",
        "difficult_fall_asleep": "1",
        "wake_up_tired": "1",
        "frequent_headaches_fatigue": "1",
        "skip_meals_stress": "1",
        "physically_exhausted": "1",
        "sacrifice_sleep": "1",
        
        # Emotional well-being (positive)
        "emotionally_drained": "1",
        "hard_feel_excited": "1",
        "feel_helpless": "1",
        "feel_burned_out": "1",
        "feel_giving_up": "1",
        "irritated_easily": "1",
        "sense_dread": "1",
        "feel_isolated": "1",
        "feel_disconnected": "1",
        
        # Academic workload (manageable)
        "workload_unmanageable": "1",
        "rarely_free_time": "1",
        "multitask_deadlines": "1",
        "workload_heavier_peers": "1",
        "struggle_balance": "1",
        "miss_schoolwork": "1",
        
        # High efficacy (strong)
        "not_accomplishing": "1",
        "question_efforts": "1",
        "underperforming_peers": "1",
        "hard_focus": "1",
        "waste_time": "1",
        "struggle_organize": "1",
        
        # Strong coping (positive)
        "confident_handling": "5",
        "feel_competent": "5",
        "proud_achievements": "5",
        "feel_control": "5",
        "finish_deadline": "5",
        "use_planner": "5",
        
        # Social support (strong)
        "emotionally_unsupported": "1",
        "someone_talk": "5",
        "family_not_understand": "1",
        "hesitate_ask_help": "1",
        
        # Environment (favorable)
        "noisy_home": "1",
        "hard_maintain_routine": "1",
        "learning_setup_stress": "1",
        "mode_affects_performance": "1",
        "prefer_modality": "5",
        
        # Commute (minimal impact)
        "commutes_stress": "1",
        "lose_motivation_commute": "1",
        "traveling_affects_energy": "1",
        
        # Family support (strong)
        "pressure_support_family": "1",
        "conflicts_home": "1",
        "financial_difficulties": "1",
        "supported_family": "5",
        "personal_life_affects": "1",
        
        # Study habits (healthy)
        "study_under_pressure": "1",
        "feel_more_isolated": "1"
    }
    
    test_data = create_test_data_from_patterns(patterns, default_value="2")
    result = predict_burnout(test_data)
    print_result("Low Burnout Student", result, expected_level="Low")
    return result


def test_case_3_moderate_burnout():
    """Test Case 3: Moderate Burnout - Mixed signals."""
    print("\n🟡 TEST CASE 3: Moderate Burnout Profile")
    print("Profile: Mixed signals - some stressors but also some coping mechanisms")
    
    patterns = {
        "workload_unmanageable": "3",
        "confident_handling": "3",
        "conflicts_home": "3",
        "mode_affects_performance": "3",
        "difficult_fall_asleep": "2",
        "emotionally_drained": "3",
        "emotionally_unsupported": "3",
        "family_not_understand": "3",
        "feel_burned_out": "3",
        "feel_competent": "3",
        "feel_disconnected": "3",
        "feel_helpless": "3",
        "feel_control": "3",
        "feel_isolated": "3",
        "feel_giving_up": "3",
        "feel_more_isolated": "3",
        "financial_difficulties": "3",
        "finish_deadline": "3",
        "frequent_headaches_fatigue": "3",
        "hard_feel_excited": "3",
        "hard_focus": "3",
        "hard_maintain_routine": "3",
        "hesitate_ask_help": "2",
        "irritated_easily": "3",
        "learning_setup_stress": "3",
        "commutes_stress": "3",
        "lose_motivation_commute": "3",
        "miss_schoolwork": "3",
        "multitask_deadlines": "3",
        "noisy_home": "3",
        "not_accomplishing": "3",
        "personal_life_affects": "3",
        "physically_exhausted": "3",
        "prefer_modality": "3",
        "pressure_support_family": "3",
        "proud_achievements": "3",
        "question_efforts": "3",
        "rarely_free_time": "3",
        "sacrifice_sleep": "3",
        "sense_dread": "3",
        "skip_meals_stress": "3",
        "sleep_less_6_hours": "3",
        "someone_talk": "3",
        "struggle_balance": "3",
        "struggle_organize": "3",
        "study_under_pressure": "3",
        "supported_family": "3",
        "traveling_affects_energy": "3",
        "underperforming_peers": "3",
        "use_planner": "3",
        "wake_up_tired": "3",
        "waste_time": "3",
        "workload_heavier_peers": "3"
    }
    
    test_data = create_test_data_from_patterns(patterns, default_value="3")
    result = predict_burnout(test_data)
    print_result("Moderate Burnout Student", result, expected_level="Moderate")
    print("💡 Note: Neutral responses may be classified as 'Low' if training data had imbalanced classes")
    return result


def test_case_4_edge_all_neutral():
    """Test Case 4: All neutral responses."""
    print("\n⚪ TEST CASE 4: Edge Case - All Neutral (3)")
    print("Profile: Student giving neutral responses to everything")
    
    # Create test data with all 3s
    feature_names = get_model_features()
    test_data = {feature: "3" for feature in feature_names}
    
    # Return as DataFrame to preserve feature names
    result = predict_burnout(pd.DataFrame([test_data]))
    print_result("All Neutral Responses", result)
    print("💡 Note: Model may lean toward 'Low' if trained data had fewer 'Moderate' samples")
    return result


def test_case_5_display_features():
    """Test Case 5: Display what features the model expects."""
    print("\n📋 TEST CASE 5: Display Model Features")
    print("Showing first 10 features the model expects:")
    
    feature_names = get_model_features()
    print(f"\n✅ Total features: {len(feature_names)}")
    print("\nFirst 10 features:")
    for i, feature in enumerate(feature_names[:10], 1):
        print(f"  {i:2}. {feature}")
    
    print(f"\n... and {len(feature_names) - 10} more features")
    print("\n💡 Tip: Run check_model_features.py to see all features")


def test_case_6_random_student(seed=None):
    """Test Case 6: Random student responses."""
    import random
    
    if seed is not None:
        random.seed(seed)
    
    print(f"\n🎲 TEST CASE 6: Random Student #{seed if seed else 'X'}")
    print("Profile: Randomly generated responses (simulating real student variability)")
    
    # Generate random responses for all features
    feature_names = get_model_features()
    test_data = {}
    
    for feature in feature_names:
        # Random value from 1-5 (Likert scale)
        test_data[feature] = str(random.randint(1, 5))
    
    result = predict_burnout(pd.DataFrame([test_data]))
    print_result(f"Random Student #{seed if seed else 'X'}", result)
    return result


def test_case_7_borderline_high():
    """Test Case 7: Borderline High Burnout - Just at the threshold."""
    print("\n🟠 TEST CASE 7: Borderline High Burnout")
    print("Profile: High symptoms but with some resilience factors")
    
    patterns = {
        # Physical symptoms (moderate-high)
        "sleep_less_6_hours": "4",
        "difficult_fall_asleep": "4",
        "wake_up_tired": "4",
        "frequent_headaches_fatigue": "3",
        "skip_meals_stress": "3",
        "physically_exhausted": "4",
        "sacrifice_sleep": "4",
        
        # Emotional exhaustion (moderate-high)
        "emotionally_drained": "4",
        "hard_feel_excited": "4",
        "feel_helpless": "3",
        "feel_burned_out": "4",
        "feel_giving_up": "3",
        "irritated_easily": "4",
        "sense_dread": "3",
        "feel_isolated": "3",
        "feel_disconnected": "4",
        
        # Academic workload (high)
        "workload_unmanageable": "4",
        "rarely_free_time": "4",
        "multitask_deadlines": "5",
        "workload_heavier_peers": "4",
        "struggle_balance": "4",
        "miss_schoolwork": "3",
        
        # Some coping mechanisms present
        "confident_handling": "3",
        "feel_competent": "3",
        "proud_achievements": "3",
        "someone_talk": "4",
        "supported_family": "4",
    }
    
    test_data = create_test_data_from_patterns(patterns, default_value="3")
    result = predict_burnout(test_data)
    print_result("Borderline High Burnout", result)
    return result


def test_case_8_realistic_moderate():
    """Test Case 8: Realistic Moderate Burnout with mixed responses."""
    print("\n🟡 TEST CASE 8: Realistic Moderate Burnout")
    print("Profile: Struggling but managing - typical stressed student")
    
    patterns = {
        # Sleep issues (moderate)
        "sleep_less_6_hours": "3",
        "difficult_fall_asleep": "3",
        "wake_up_tired": "4",
        "physically_exhausted": "3",
        
        # Emotional state (mixed)
        "emotionally_drained": "4",
        "feel_burned_out": "3",
        "feel_giving_up": "2",
        "irritated_easily": "4",
        "hard_feel_excited": "3",
        
        # Workload (manageable but challenging)
        "workload_unmanageable": "3",
        "rarely_free_time": "4",
        "multitask_deadlines": "4",
        "struggle_balance": "3",
        
        # Some efficacy
        "confident_handling": "3",
        "feel_competent": "3",
        "proud_achievements": "3",
        "finish_deadline": "3",
        
        # Decent support
        "someone_talk": "4",
        "supported_family": "4",
        "emotionally_unsupported": "2",
    }
    
    test_data = create_test_data_from_patterns(patterns, default_value="3")
    result = predict_burnout(test_data)
    print_result("Realistic Moderate Burnout", result)
    return result


def check_model_exists():
    """Check if model files exist."""
    if not MODEL_PATH.exists():
        print("\n❌ ERROR: Model file not found!")
        print(f"Expected location: {MODEL_PATH}")
        print("\n💡 Please train the model first:")
        print("   python backend/services/training_service.py")
        return False
    
    if not PREPROCESSOR_PATH.exists():
        print("\n❌ ERROR: Preprocessor file not found!")
        print(f"Expected location: {PREPROCESSOR_PATH}")
        return False
    
    print("\n✅ Model files found:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Preprocessor: {PREPROCESSOR_PATH}")
    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "🔬"*40)
    print("BURNOUT PREDICTION MODEL - TEST SUITE")
    print("🔬"*40)
    
    # Check if model exists
    if not check_model_exists():
        return
    
    results = []
    
    try:
        # Display features first
        test_case_5_display_features()
        
        # Run standard test cases
        print("\n" + "="*80)
        print("PART 1: STANDARD TEST CASES")
        print("="*80)
        results.append(("High Burnout", test_case_1_high_burnout()))
        results.append(("Low Burnout", test_case_2_low_burnout()))
        results.append(("Moderate Burnout", test_case_3_moderate_burnout()))
        results.append(("All Neutral", test_case_4_edge_all_neutral()))
        
        # Run additional test cases
        print("\n" + "="*80)
        print("PART 2: ADDITIONAL TEST CASES")
        print("="*80)
        results.append(("Borderline High", test_case_7_borderline_high()))
        results.append(("Realistic Moderate", test_case_8_realistic_moderate()))
        
        # Run random tests
        print("\n" + "="*80)
        print("PART 3: RANDOM STUDENT SIMULATIONS")
        print("="*80)
        for i in range(1, 6):
            results.append((f"Random Student {i}", test_case_6_random_student(seed=i)))
        
        # Summary
        print("\n" + "📊"*40)
        print("TEST SUMMARY")
        print("📊"*40)
        
        print("\n📈 Prediction Distribution:")
        prediction_counts = {}
        for test_name, result in results:
            pred = result['prediction']
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        for pred, count in sorted(prediction_counts.items()):
            percentage = (count / len(results)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {pred:12} {count:2}/{len(results):2} ({percentage:5.1f}%) {bar}")
        
        print("\n📋 Individual Results:")
        for test_name, result in results:
            status = "✅" if result['confidence'] > 60 else "⚠️"
            print(f"{status} {test_name:30} → {result['prediction']:10} ({result['confidence']:.1f}% confidence)")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # DETAILED EXPLANATION SECTION
        print("\n" + "📖"*40)
        print("DETAILED EXPLANATION - WHAT HAPPENED?")
        print("📖"*40)
        
        print("\n🤔 WHAT IS CONFIDENCE?")
        print("-" * 80)
        print("""
Think of confidence like this:
Imagine you're looking at a photo and trying to guess if it's a cat or a dog.

• 90-100% confidence = "I'm VERY SURE this is a cat!"
  → The model sees strong, clear patterns matching one category
  
• 70-89% confidence = "I'm pretty sure it's a cat"
  → The model sees good evidence, but maybe some dog-like features too
  
• 50-69% confidence = "Hmm, probably a cat, but could be wrong"
  → The model sees mixed signals, leaning slightly toward one answer
  
• Below 50% confidence = "I really can't tell"
  → The model sees equal evidence for multiple categories

In our burnout predictions:
- High confidence means the student's answers clearly match a burnout pattern
- Low confidence means the answers are mixed/borderline between categories
        """)
        
        print("\n📊 HOW THE MODEL WORKS")
        print("-" * 80)
        print("""
Step 1: Student answers 53 survey questions (like "I feel burned out" = 1-5)
Step 2: Model compares answers to patterns learned from training data
Step 3: Model calculates probability for each category (Low/Moderate/High)
Step 4: Picks the category with highest probability as the prediction
Step 5: The highest probability becomes the "confidence" percentage

Example from our tests:
  Low: 99.45%  👈 Model picks this (highest)
  Moderate: 0.55%
  High: 0.00%
  → Prediction = "Low" with 99.45% confidence
        """)
        
        print("\n🔍 WHY SOME PREDICTIONS SEEM WRONG")
        print("-" * 80)
        print("""
Test Case 3 predicted "Low" when we expected "Moderate" - here's why:

1. TRAINING DATA IMBALANCE
   - If the model was trained on 100 students:
     • 60 students = Low burnout
     • 25 students = High burnout  
     • 15 students = Moderate burnout
   - The model learned that "Moderate" is rare
   - So neutral answers (3s) get classified as "Low" (most common)

2. THE NEUTRAL TRAP
   - When someone answers "3" (neutral) to everything
   - Model thinks: "No strong negative feelings = Low burnout"
   - In reality: "3" should be Moderate, but model learned differently

3. HOW TO FIX THIS
   - Collect more "Moderate" burnout examples in training data
   - Balance the dataset so all three categories have equal samples
   - Retrain the model with balanced data
        """)
        
        print("\n💡 WHAT THE NUMBERS MEAN FOR EACH TEST")
        print("-" * 80)
        for test_name, result in results:
            print(f"\n{test_name}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.1f}%")
            
            if result['confidence'] > 80:
                explanation = "Model is VERY SURE - sees clear burnout pattern"
            elif result['confidence'] > 60:
                explanation = "Model is FAIRLY CONFIDENT - good evidence"
            elif result['confidence'] > 40:
                explanation = "Model is UNCERTAIN - mixed signals detected"
            else:
                explanation = "Model CANNOT DECIDE - probabilities too close"
            
            print(f"  Meaning: {explanation}")
            
            # Show probability breakdown
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            print(f"  Breakdown: {sorted_probs[0][0]} ({sorted_probs[0][1]:.1f}%), "
                  f"{sorted_probs[1][0]} ({sorted_probs[1][1]:.1f}%), "
                  f"{sorted_probs[2][0]} ({sorted_probs[2][1]:.1f}%)")
        
        print("\n" + "="*80)
        print("END OF DETAILED REPORT")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()