
import pandas as pd

# Load data
df = pd.read_csv("data/behaviour_data.csv")

# -------- FEATURE ENGINEERING --------

# Sleep efficiency score
df["sleep_efficiency"] = df["sleep_hours"] / (df["screen_time"] + 1)

# Productivity score
df["productivity_score"] = df["work_hours"] * df["mood"]

# Digital overuse flag
df["digital_overuse"] = df["screen_time"].apply(
    lambda x: 1 if x > 6 else 0
)

# Physical activity level
df["activity_level"] = pd.cut(
    df["steps"],
    bins=[0, 3000, 7000, 15000],
    labels=["low", "medium", "high"]
)

print("Feature engineering completed!")
print(df.head())

from sklearn.preprocessing import StandardScaler

# -------- PREPARE DATA FOR CLUSTERING --------

# Select features for clustering
features = df[
    [
        "sleep_hours",
        "screen_time",
        "work_hours",
        "steps",
        "mood",
        "sleep_efficiency",
        "productivity_score",
        "digital_overuse"
    ]
]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("\nData prepared for clustering")
print("Shape of scaled data:", scaled_features.shape)
