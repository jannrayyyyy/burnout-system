#!/usr/bin/env python3
"""
Generate the correct field mapping between frontend and model
Save as: backend/services/generate_mapping.py
Run: python backend/services/generate_mapping.py
"""

import joblib
from pathlib import Path
import re

# Load preprocessor to get actual feature names
PREPROCESSOR_PATH = Path("models/preprocessor_latest.pkl")
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_names = preprocessor['feature_names']

print("\n" + "="*80)
print("GENERATING CORRECT FIELD MAPPING")
print("="*80)

print(f"\n✅ Model expects {len(feature_names)} features")

# Your current SHORT mapping from predict.py
CURRENT_SHORT_MAPPING = {
    "sleep_less_than_6_hours": "sleep_less_6_hours",
    "difficulty_fall_asleep": "difficult_fall_asleep",
    "wake_up_tired": "wake_up_tired",
    "frequent_headaches_or_fatigue": "frequent_headaches_fatigue",
    "skip_meals_due_to_stress": "skip_meals_stress",
    "physically_exhausted": "physically_exhausted",
    "multitask_deadlines": "multitask_deadlines",
    "study_under_pressure": "study_under_pressure",
    "academic_workload_unmanageable": "workload_unmanageable",
    "rarely_have_free_time": "rarely_free_time",
    "struggle_organize_tasks": "struggle_organize",
    "use_planner": "use_planner",
    "sacrifice_sleep": "sacrifice_sleep",
    "workload_heavier_than_peers": "workload_heavier_peers",
    "emotionally_drained_end_of_day": "emotionally_drained",
    "hard_to_feel_excited": "hard_feel_excited",
    "feel_helpless": "feel_helpless",
    "feel_burned_out": "feel_burned_out",
    "feel_like_giving_up": "feel_giving_up",
    "irritated_easily": "irritated_easily",
    "sense_of_dread": "sense_dread",
    "proud_of_achievements": "proud_achievements",
    "not_accomplishing_anything": "not_accomplishing",
    "confident_handling_challenges": "confident_handling",
    "question_efforts": "question_efforts",
    "feel_competent": "feel_competent",
    "underperforming_peers": "underperforming_peers",
    "struggle_balance_responsibilities": "struggle_balance",
    "waste_time": "waste_time",
    "hard_to_maintain_routine": "hard_maintain_routine",
    "finish_before_deadline": "finish_deadline",
    "feel_in_control": "feel_control",
    "supported_by_family": "supported_family",
    "feel_isolated": "feel_isolated",
    "hesitate_ask_help": "hesitate_ask_help",
    "someone_to_talk": "someone_talk",
    "feel_disconnected": "feel_disconnected",
    "financial_difficulties": "financial_difficulties",
    "pressure_support_family": "pressure_support_family",
    "conflicts_at_home": "conflicts_home",
    "noisy_home_environment": "noisy_home",
    "emotionally_unsupported": "emotionally_unsupported",
    "miss_schoolwork": "miss_schoolwork",
    "family_does_not_understand": "family_not_understand",
    "personal_life_affects_academics": "personal_life_affects",
    "traveling_affects_energy": "traveling_affects_energy",
    "long_commutes_stress": "commutes_stress",
    "lose_motivation_commute": "lose_motivation_commute",
    "hard_to_focus": "hard_focus",
    "learning_setup_stress": "learning_setup_stress",
    "feel_more_isolated": "feel_more_isolated",
    "prefer_current_modality": "prefer_modality",
    "current_mode_affects_performance": "mode_affects_performance",
}


def normalize_for_matching(text):
    """Normalize text for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text


def find_best_match(short_key, full_features):
    """Find the best matching full feature name for a short key."""
    normalized_short = normalize_for_matching(short_key)
    
    # Try exact substring match first
    for full_feature in full_features:
        normalized_full = normalize_for_matching(full_feature)
        if normalized_short in normalized_full or normalized_full in normalized_short:
            return full_feature
    
    # Try keyword matching
    short_words = set(normalized_short.split('_'))
    best_match = None
    best_score = 0
    
    for full_feature in full_features:
        normalized_full = normalize_for_matching(full_feature)
        full_words = set(normalized_full.split('_'))
        
        # Count matching words
        matching_words = short_words & full_words
        score = len(matching_words)
        
        if score > best_score:
            best_score = score
            best_match = full_feature
    
    return best_match if best_score > 0 else None


print("\n" + "="*80)
print("MAPPING ANALYSIS")
print("="*80)

correct_mapping = {}
unmatched = []

for frontend_key, short_key in CURRENT_SHORT_MAPPING.items():
    # Try to find matching full feature name
    best_match = find_best_match(short_key, feature_names)
    
    if best_match:
        correct_mapping[frontend_key] = best_match
        print(f"✅ {frontend_key}")
        print(f"   → {best_match}")
    else:
        unmatched.append((frontend_key, short_key))
        print(f"❌ {frontend_key} (short: {short_key})")
        print(f"   → NO MATCH FOUND")

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"✅ Matched: {len(correct_mapping)}/{len(CURRENT_SHORT_MAPPING)}")
print(f"❌ Unmatched: {len(unmatched)}")

if unmatched:
    print(f"\n⚠️  Unmatched fields:")
    for frontend_key, short_key in unmatched:
        print(f"   - {frontend_key} ({short_key})")

# Generate the corrected mapping code
print(f"\n{'='*80}")
print("CORRECTED FIELD_MAPPING FOR predict.py")
print(f"{'='*80}\n")

print("# Replace FIELD_MAPPING in backend/routers/predict.py with this:")
print("\nFIELD_MAPPING = {")
for frontend_key, full_feature in sorted(correct_mapping.items()):
    print(f'    "{frontend_key}": "{full_feature}",')
print("}")

# Save to file
output_file = Path("corrected_field_mapping.py")
with open(output_file, 'w') as f:
    f.write("# Corrected FIELD_MAPPING for backend/routers/predict.py\n")
    f.write("# Replace the existing FIELD_MAPPING with this:\n\n")
    f.write("FIELD_MAPPING = {\n")
    for frontend_key, full_feature in sorted(correct_mapping.items()):
        f.write(f'    "{frontend_key}": "{full_feature}",\n')
    f.write("}\n")

print(f"\n✅ Saved corrected mapping to: {output_file}")
print(f"\n💡 Next steps:")
print(f"   1. Review the mapping above")
print(f"   2. Copy the corrected FIELD_MAPPING")
print(f"   3. Replace it in backend/routers/predict.py")
print(f"   4. Restart your API server")
print(f"   5. Test again - results should match test_prediction.py!")