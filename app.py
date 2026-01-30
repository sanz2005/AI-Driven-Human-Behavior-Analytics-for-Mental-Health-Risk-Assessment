import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from src.pdf_report import generate_pdf
from src.ai_coach import ai_coach


# ---------- Load Data ----------
df = pd.read_csv("data/behaviour_data.csv")

# ---------- Phase-A: History Setup ----------
HISTORY_FILE = "data/user_history.csv"

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "sleep","screen","work","steps",
        "mood","stress","digital","wellness","burnout"
    ]).to_csv(HISTORY_FILE, index=False)

# ---------- Feature Engineering ----------
df["sleep_efficiency"] = df["sleep_hours"] / (df["screen_time"] + 1)
df["productivity_score"] = df["work_hours"] * df["mood"]
df["digital_overuse"] = df["screen_time"].apply(lambda x: 1 if x > 6 else 0)

# ---------- Mental Health Scores ----------
df["stress_index"] = (df["work_hours"] * df["screen_time"]) / (df["sleep_hours"] + 1)
df["digital_addiction"] = df["screen_time"] / (df["steps"] / 1000 + 1)
df["wellness_score"] = df["sleep_hours"] + (df["steps"] / 1000) + df["mood"] - df["screen_time"]
df["burnout_risk"] = (df["work_hours"] * 2 + df["screen_time"]) / (df["sleep_hours"] + df["mood"])

# ---------- Clustering ----------
cluster_features = df[
    ["sleep_hours","screen_time","work_hours","steps",
     "sleep_efficiency","productivity_score","digital_overuse"]
]

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled)

# ---------- Mood Prediction ----------
X = df[["sleep_hours","screen_time","work_hours","steps"]]
y = df["mood"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Explainable AI
importance = model.feature_importances_

# ---------- UI ----------
st.set_page_config(layout="wide")
st.title("🧠 AI-Driven Human Behavior Analytics for Mental Health Risk Assessment")

st.sidebar.header("📥 Enter Your Daily Data")
sleep = st.sidebar.slider("Sleep Hours", 3.0, 10.0, 7.0)
screen = st.sidebar.slider("Screen Time (hrs)", 1.0, 12.0, 5.0)
work = st.sidebar.slider("Work Hours", 0.0, 12.0, 6.0)
steps = st.sidebar.slider("Steps", 1000, 15000, 7000)

# ---------- Prediction ----------
user_df = pd.DataFrame([[sleep, screen, work, steps]],
                       columns=["sleep_hours","screen_time","work_hours","steps"])
pred_mood = model.predict(user_df)[0]

sleep_eff = sleep / (screen + 1)
prod_score = work * pred_mood
digital = 1 if screen > 6 else 0
cluster_input = scaler.transform([[sleep, screen, work, steps, sleep_eff, prod_score, digital]])
cluster = kmeans.predict(cluster_input)[0]

# Raw scores
stress = (work * screen) / (sleep + 1)
digital_add = screen / (steps/1000 + 1)
wellness = sleep + (steps/1000) + pred_mood - screen
burnout = (work * 2 + screen) / (sleep + pred_mood)

# ---------- Normalize to 0–100 ----------
def normalize(val, min_val, max_val):
    return ((val - min_val) / (max_val - min_val)) * 100

stress_pct = normalize(stress, df["stress_index"].min(), df["stress_index"].max())
digital_pct = normalize(digital_add, df["digital_addiction"].min(), df["digital_addiction"].max())
wellness_pct = normalize(wellness, df["wellness_score"].min(), df["wellness_score"].max())
burnout_pct = normalize(burnout, df["burnout_risk"].min(), df["burnout_risk"].max())
mood_pct = (pred_mood / 5) * 100

# ---------- Phase-A: Save to History ----------
try:
    history = pd.read_csv(HISTORY_FILE)
except:
    history = pd.DataFrame(columns=[
        "sleep","screen","work","steps",
        "mood","stress","digital","wellness","burnout"
    ])

new_row = pd.DataFrame([[sleep, screen, work, steps,
                         pred_mood, stress_pct,
                         digital_pct, wellness_pct, burnout_pct]],
                       columns=history.columns)

history = pd.concat([history, new_row], ignore_index=True)
history.to_csv(HISTORY_FILE, index=False)

# ---------- Status Labels ----------
def label(value, good_high=True):
    if good_high:
        if value > 70: return "🟢 Healthy"
        elif value > 40: return "🟡 Moderate"
        else: return "🔴 Poor"
    else:
        if value < 30: return "🟢 Safe"
        elif value < 60: return "🟡 Warning"
        else: return "🔴 Dangerous"

# ---------- Display ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 AI Prediction")
    st.write("Predicted Mood:", round(pred_mood,2), "/ 5")
    st.progress(int(mood_pct))
    st.write("Mood Level:", round(mood_pct,1), "%")

    # -------- Added Lifestyle Type Block --------
    cluster_map = {
        0: ("📱 Digitally Overloaded",
            "High screen time and low physical activity. Risk of digital fatigue."),
        1: ("⚖ Balanced Performer",
            "Good balance of sleep, work, and activity. Healthy lifestyle."),
        2: ("🔥 Burnout Risk",
            "High work and screen time with low recovery. Stress prone."),
        3: ("🛋 Low Activity",
            "Low physical movement and low energy. Improve activity levels.")
    }

    name, desc = cluster_map[int(cluster)]

    st.subheader("🧬 Your Lifestyle Type")
    st.success(name)
    st.write(desc)
    # -------------------------------------------

with col2:
    st.subheader("🧠 Mental Health Scores (0–100)")
    st.write("Stress")
    st.progress(int(stress_pct))
    st.write(label(stress_pct, False))

    st.write("Digital Addiction")
    st.progress(int(digital_pct))
    st.write(label(digital_pct, False))

    st.write("Wellness")
    st.progress(int(wellness_pct))
    st.write(label(wellness_pct, True))

    st.write("Burnout Risk")
    st.progress(int(burnout_pct))
    st.write(label(burnout_pct, False))

# ---------- Phase-A: Trend Charts ----------
st.subheader("📈 Your Health Trends")

history = pd.read_csv(HISTORY_FILE)

if len(history) > 2:
    st.line_chart(history[["mood","stress","burnout","wellness"]])
else:
    st.info("Use the app multiple times to generate trends")

# ---------- Explainable AI ----------
st.subheader("🔍 Why This Mood?")
features = ["Sleep","Screen Time","Work","Steps"]
for f, v in zip(features, importance):
    st.write(f, "contributes", round(v*100,1), "% to your mood")

# ---------- PHASE B: PDF Health Report ----------
st.subheader("🧾 Download Your Health Report")

history = pd.read_csv(HISTORY_FILE)
latest = history.iloc[-1]

summary = {
    "Predicted Mood (0–100)": f"{round((latest['mood']/5)*100,1)}%",
    "Stress Level": f"{round(latest['stress'],1)}%",
    "Digital Addiction": f"{round(latest['digital'],1)}%",
    "Wellness Score": f"{round(latest['wellness'],1)}%",
    "Burnout Risk": f"{round(latest['burnout'],1)}%",
}

trends = {
    "Mood trend": round(history["mood"].tail(5).mean(),2),
    "Stress trend": round(history["stress"].tail(5).mean(),2),
    "Burnout trend": round(history["burnout"].tail(5).mean(),2),
    "Wellness trend": round(history["wellness"].tail(5).mean(),2),
}

if st.button("📥 Generate Health Report (PDF)"):
    pdf_path = "data/health_report.pdf"
    generate_pdf(pdf_path, summary, trends)
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF", f, file_name="health_report.pdf")

# ---------- PHASE C: AI Coach ----------
st.subheader("🤖 AI Health Coach")

history = pd.read_csv(HISTORY_FILE)

if len(history) > 1:
    advice = ai_coach(history)
    for tip in advice:
        st.write("•", tip)
else:
    st.info("Use the app a few times to activate AI Coach")
