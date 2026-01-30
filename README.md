🧠 AI-Driven Human Behavior Analytics for Mental Health Risk Assessment

An AI-powered human behavior analytics system that leverages machine learning, behavioral clustering, and explainable AI to assess mental health risks based on daily lifestyle patterns. The system predicts mood, identifies burnout and stress risk, tracks long-term trends, and provides personalized insights through an interactive Streamlit dashboard.

🚀 Project Overview

This project analyzes daily behavioral data such as sleep duration, screen time, work hours, and physical activity to uncover hidden behavior patterns and mental health indicators. By combining supervised and unsupervised machine learning techniques, the system provides:

- Mood prediction
- Mental health risk assessment
- Behavioral segmentation
- Trend analysis over time
- Explainable AI insights
- Personalized AI health coaching
- Downloadable PDF health reports

The goal is to support preventive, data-driven mental health monitoring and lifestyle optimization.

🎯 Key Features 

🔮 Mood Prediction using Random Forest regression

🧠 Mental Health Scoring (Stress, Wellness, Burnout, Digital Addiction)

🧬 Behavioral Clustering using K-Means

📊 Trend Analysis from historical user data

🔍 Explainable AI showing feature impact on mood

🤖 AI Health Coach with personalized recommendations

📄 PDF Health Report Generation

🖥️ Interactive Streamlit Dashboard

🛠️ Tech Stack
Programming Language: Python

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-learn

Visualization & UI: Streamlit

Explainability: Feature importance (Explainable AI)

📂 Project Structure

Human-Behavior-Analytics-Mental-Health/
│
├── app.py                     # Main Streamlit application
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── src/                       # Core logic used by the app
│   ├── ai_coach.py            # AI health coach logic
│   ├── pdf_report.py          # PDF report generation
│   ├── prediction.py          # Mood prediction model
│   ├── mental_health_scores.py# Mental health score calculations
│   └── explainable_ai.py      # Explainable AI logic
│
├── experiments/               # Research and experimentation scripts
│   ├── eda.py                 # Exploratory data analysis
│   ├── clustering.py          # Behavioral clustering experiments
│   ├── cluster_optimization.py# Cluster tuning and evaluation
│   ├── time_series_prediction.py # Time-series and trend analysis
│   ├── behavior_fingerprint.py# Behavioral pattern extraction
│   └── feature_engineering.py # Feature engineering experiments
│
├── data/
│   └── behaviour_data.csv     # Dataset used for training and analysis
│
└── assets/
    ├── dashboard.png          # Dashboard screenshots
    ├── scores.png             # Score visualization screenshots
    └── coach.png              # AI coach output screenshots
    
📊 How the System Works

1. User Input
Users enter daily lifestyle data (sleep, screen time, work hours, steps).

2. Feature Engineering
Behavioral indicators such as sleep efficiency, productivity score, and digital overuse are computed.

3. Machine Learning Models

4. Regression Model: Predicts mood score

5. Clustering Model: Segments users into lifestyle behavior groups

6. Mental Health Scoring
Scores are normalized to a 0–100 scale for easy interpretation.

7. Explainable AI
Shows how each lifestyle factor contributes to the predicted mood.

8. Trend Tracking
Historical data is stored and visualized to show long-term health trends.

9. AI Health Coach
Generates personalized lifestyle recommendations based on user patterns.

10. Report Generation
Users can download a detailed PDF health report.

11. Reporting: PDF generation

12. Version Control: GitHub

▶️ How to Run the Project

1️) Install Dependencies
   pip install -r requirements.txt

2) Run the Streamlit App
   streamlit run app.py

   
