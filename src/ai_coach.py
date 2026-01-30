def ai_coach(history):
    latest = history.iloc[-1]

    stress = latest["stress"]
    burnout = latest["burnout"]
    wellness = latest["wellness"]
    digital = latest["digital"]
    mood = latest["mood"]

    advice = []

    if burnout > 60:
        advice.append("⚠ You are at high burnout risk. Try reducing work hours and improving sleep.")
    if stress > 60:
        advice.append("High stress detected. Consider breaks, relaxation, and reducing screen exposure.")
    if digital > 60:
        advice.append("You may be digitally overloaded. Try limiting screen time and increasing physical activity.")
    if wellness < 40:
        advice.append("Your wellness is low. Focus on sleep, movement, and balanced daily routines.")
    if mood < 50:
        advice.append("Your mood has been low. Improving sleep and reducing screen time could help.")

    if not advice:
        advice.append("You are doing well! Maintain your current healthy lifestyle.")

    return advice

