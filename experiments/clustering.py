import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/behaviour_data.csv")

# -------- FEATURE ENGINEERING (same as before) --------
df["sleep_efficiency"] = df["sleep_hours"] / (df["screen_time"] + 1)
df["productivity_score"] = df["work_hours"] * df["mood"]
df["digital_overuse"] = df["screen_time"].apply(lambda x: 1 if x > 6 else 0)

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

# -------- KMEANS CLUSTERING --------
kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_features)

print("Clustering completed!")
print(df[["person_id", "cluster"]].head())

# -------- CLUSTER INTERPRETATION --------

cluster_summary = df.groupby("cluster")[[
    "sleep_hours",
    "screen_time",
    "work_hours",
    "steps",
    "mood",
    "sleep_efficiency",
    "productivity_score",
    "digital_overuse"
]].mean()

print("\nCluster Summary (Mean Values):")
print(cluster_summary)

from sklearn.cluster import DBSCAN

# -------- DBSCAN CLUSTERING --------

dbscan = DBSCAN(eps=1.2, min_samples=10)
df["dbscan_cluster"] = dbscan.fit_predict(scaled_features)

print("\nDBSCAN clustering completed!")
print(df["dbscan_cluster"].value_counts())

