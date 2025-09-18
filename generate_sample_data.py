# generate_sample_data.py
import pandas as pd
import numpy as np
from pathlib import Path

Path("data").mkdir(exist_ok=True)

N = 500
np.random.seed(42)
df = pd.DataFrame({
    "age": np.random.randint(17, 25, N),
    "gender": np.random.choice(["Male","Female","Other"], N, p=[0.45,0.45,0.10]),
    "year_level": np.random.randint(1, 6, N),
    "gwa": np.round(np.random.uniform(1.0, 5.0, N), 2),
    "num_subjects": np.random.randint(3,10, N),
    "hours_online": np.round(np.random.uniform(0, 8, N),1),
    "study_hours": np.round(np.random.uniform(0, 12, N),1),
    "sleep_hours": np.round(np.random.uniform(3, 9, N),1),
    "perceived_stress": np.random.randint(1,6,N),
    "procrastination": np.random.randint(1,6,N),
    "motivation": np.random.randint(1,6,N),
})

# small rule to create burnout_level for demo purposes:
sleep_penalty = pd.cut(df["sleep_hours"], bins=[-1,4,6,100], labels=[2,1,0]).astype(int)
score = df["perceived_stress"] + (6 - df["motivation"]) + sleep_penalty
df["burnout_level"] = pd.cut(score, bins=[-1,3,6,100], labels=["Low","Moderate","High"])

df.to_csv("data/surveys.csv", index=False)
print("Wrote data/surveys.csv with", len(df), "rows")
