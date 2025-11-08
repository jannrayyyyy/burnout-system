#!/usr/bin/env python3
"""
Fixed test to show the feature name mapping issue
Save as: backend/services/show_mapping.py
Run: python backend/services/show_mapping.py
"""

import joblib
from pathlib import Path

# Paths
PREPROCESSOR_PATH = Path("models/preprocessor_latest.pkl")

print("\n" + "="*80)
print("FEATURE NAME ANALYSIS")
print("="*80)

# Load preprocessor
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_names = preprocessor['feature_names']

print(f"\n✅ Model expects {len(feature_names)} features")
print(f"\n📋 Here are the actual feature names the model expects:")
print("="*80)

for i, feature in enumerate(feature_names, 1):
    print(f"{i:2}. {feature}")

print("\n" + "="*80)
print("THE PROBLEM")
print("="*80)

print("""
Your test_prediction.py uses SHORT names like:
  - sleep_less_6_hours
  - difficult_fall_asleep
  - wake_up_tired

But your model expects FULL question text like:
  - sleep_patterns_and_physical_health_i_usually_get_less_than_6_hours_of_sleep_on_school_nights.
  - sleep_patterns_and_physical_health_i_find_it_difficult_to_fall_asleep_because_of_academic_stress.
  - sleep_patterns_and_physical_health_i_often_wake_up_feeling_tired_or_unrefreshed.

This means:
1. Your API router (predict.py) must have a mapping function
2. That converts short keys → full question text
3. THAT'S where the transformation happens
4. And THAT'S why results are different!

""")

print("="*80)
print("SOLUTION")
print("="*80)

print("""
You need to find the mapping function in your API router.

Check: backend/routers/predict.py

Look for:
  - A dictionary/mapping of short names → full questions
  - A function that transforms field names
  - Something that converts "sleep_less_6_hours" to the full question text

Share that file and I'll fix the mapping logic!
""")

print("\n💡 QUICK FIX for testing:")
print("   Your test_prediction.py needs to use the FULL feature names")
print("   OR you need the same mapping function your API uses")