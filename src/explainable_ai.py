import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/behaviour_data.csv")

# Feature engineering
df["sleep_efficiency"] = df["sleep_hours"] / (df["screen_time"] + 1)
df["productivity_score"] = df["work_hours"] * df["mood"]

X = df[[
    "sleep_hours",
    "screen_time",
    "work_hours",
    "steps",
    "sleep_efficiency",
    "productivity_score"
]]

y = df["mood"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

importances = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("Feature importance for Mood Prediction:")
print(feature_importance)

