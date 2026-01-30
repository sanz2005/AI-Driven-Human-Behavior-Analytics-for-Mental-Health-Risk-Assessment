
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/behaviour_data.csv")

# Sort by person and day
df = df.sort_values(["person_id", "day"])

# Create lag features (yesterday’s behavior)
df["sleep_prev"] = df.groupby("person_id")["sleep_hours"].shift(1)
df["screen_prev"] = df.groupby("person_id")["screen_time"].shift(1)
df["work_prev"] = df.groupby("person_id")["work_hours"].shift(1)
df["mood_prev"] = df.groupby("person_id")["mood"].shift(1)

# Remove first-day rows (NaN)
df = df.dropna()

# Features and target
X = df[
    ["sleep_prev", "screen_prev", "work_prev", "mood_prev"]
]
y = df["mood"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Time-Series Prediction Results")
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))
print("R2:", round(r2_score(y_test, y_pred), 3))
