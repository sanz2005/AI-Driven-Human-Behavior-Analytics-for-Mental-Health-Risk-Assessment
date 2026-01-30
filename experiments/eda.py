
import pandas as pd

# Load dataset
df = pd.read_csv("data/behaviour_data.csv")

# Basic inspection
print("First 5 rows:\n")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:\n")
print(df.describe())

print("\nMood distribution:")
print(df["mood"].value_counts().sort_index())

import matplotlib.pyplot as plt

# ---------- HISTOGRAMS ----------

plt.figure()
plt.hist(df["sleep_hours"], bins=20)
plt.title("Sleep Hours Distribution")
plt.xlabel("Sleep Hours")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(df["screen_time"], bins=20)
plt.title("Screen Time Distribution")
plt.xlabel("Screen Time (hrs)")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(df["work_hours"], bins=20)
plt.title("Work Hours Distribution")
plt.xlabel("Work Hours")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(df["mood"], bins=5)
plt.title("Mood Distribution")
plt.xlabel("Mood (1–5)")
plt.ylabel("Frequency")
plt.show()

import seaborn as sns

# ---------- CORRELATION MATRIX ----------

corr = df[[
    "sleep_hours",
    "screen_time",
    "work_hours",
    "steps",
    "mood"
]].corr()

plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()
