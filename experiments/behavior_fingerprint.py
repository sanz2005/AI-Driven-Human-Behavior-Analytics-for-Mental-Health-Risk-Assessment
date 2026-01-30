import pandas as pd

df = pd.read_csv("data/behaviour_data.csv")

df = df.sort_values(["person_id", "day"])

fingerprints = []

for pid in df["person_id"].unique():
    person = df[df["person_id"] == pid]

    sleep_stability = person["sleep_hours"].std()
    mood_volatility = person["mood"].std()
    avg_screen = person["screen_time"].mean()
    avg_work = person["work_hours"].mean()
    wellness = (
        person["sleep_hours"].mean() +
        person["steps"].mean() / 1000 +
        person["mood"].mean() -
        person["screen_time"].mean()
    )

    fingerprints.append([
        pid,
        round(sleep_stability,2),
        round(mood_volatility,2),
        round(avg_screen,2),
        round(avg_work,2),
        round(wellness,2)
    ])

fingerprint_df = pd.DataFrame(
    fingerprints,
    columns=[
        "person_id",
        "sleep_stability",
        "mood_volatility",
        "avg_screen_time",
        "avg_work_hours",
        "wellness_index"
    ]
)

print(fingerprint_df.head())

