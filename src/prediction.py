import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/behaviour_data.csv")

# Feature engineering (same logic as before)
df["sleep_efficiency"] = df["sleep_hours"] / (df["screen_time"] + 1)
df["productivity_score"] = df["work_hours"] * df["mood"]

# Features & target
X = df[
    [
        "sleep_hours",
        "screen_time",
        "work_hours",
        "steps",
        "sleep_efficiency",
        "productivity_score"
    ]
]
y = df["mood"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Prediction Model Results")
print("Mean Squared Error:", round(mse, 3))
print("R2 Score:", round(r2, 3))

