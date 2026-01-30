
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv("data/behaviour_data.csv")

# Feature engineering
df["sleep_efficiency"] = df["sleep_hours"] / (df["screen_time"] + 1)
df["productivity_score"] = df["work_hours"] * df["mood"]
df["digital_overuse"] = df["screen_time"].apply(lambda x: 1 if x > 6 else 0)

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

# Scale
scaled = StandardScaler().fit_transform(features)

# ---------- ELBOW METHOD ----------
inertia = []
K = range(2, 10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled)
    inertia.append(model.inertia_)

plt.figure()
plt.plot(K, inertia)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

# ---------- SILHOUETTE SCORE ----------
print("Silhouette Scores:")
for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(scaled)
    score = silhouette_score(scaled, labels)
    print(f"K={k} → Silhouette Score = {round(score, 3)}")
