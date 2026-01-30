import pandas as pd

df = pd.read_csv("data/behaviour_data.csv")

# ----- Mental Health & Lifestyle Scores -----

df["stress_index"] = (df["work_hours"] * df["screen_time"]) / (df["sleep_hours"] + 1)

df["digital_addiction"] = df["screen_time"] / (df["steps"] / 1000 + 1)

df["wellness_score"] = (
    df["sleep_hours"] +
    (df["steps"] / 1000) +
    df["mood"] -
    df["screen_time"]
)

df["burnout_risk"] = (df["work_hours"] * 2 + df["screen_time"]) / (df["sleep_hours"] + df["mood"])

print(df[[
    "sleep_hours", "screen_time", "work_hours", "steps", "mood",
    "stress_index", "digital_addiction", "wellness_score", "burnout_risk"
]].head())

