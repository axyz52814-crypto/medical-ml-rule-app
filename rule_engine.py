"""
rule_engine.py
--------------
A TRANSPARENT, rule-based risk scorer. This is the "explainable by construction"
half of the thesis: every point it assigns is a rule you can read and defend.

IMPORTANT (put this in your thesis and the app UI):
This is a SIMPLIFIED, EDUCATIONAL scorer loosely inspired by well-known clinical
risk factors (BMI categories, blood pressure, cholesterol, activity). It is NOT a
validated clinical instrument and must not be used for real medical decisions.
Its VALUE here is transparency and as a baseline to compare the ML model against.

score() returns:
  - total points (int)
  - a category string
  - a breakdown list explaining exactly where each point came from
"""


def _bmi_points(bmi):
    if bmi < 18.5:
        return 1, "Underweight (BMI < 18.5)"
    if bmi < 25:
        return 0, "Healthy weight (BMI 18.5-24.9)"
    if bmi < 30:
        return 1, "Overweight (BMI 25-29.9)"
    if bmi < 35:
        return 2, "Obese class I (BMI 30-34.9)"
    return 3, "Obese class II+ (BMI >= 35)"


def _age_points(age_bucket):
    # BRFSS Age is a 1..13 bucket; treat older buckets as higher risk.
    if age_bucket <= 3:      # ~18-34
        return 0, "Age under ~35"
    if age_bucket <= 7:      # ~35-54
        return 1, "Age ~35-54"
    if age_bucket <= 9:      # ~55-64
        return 2, "Age ~55-64"
    return 3, "Age ~65+"


def score(inputs: dict):
    """
    inputs: dict with keys BMI, HighBP, HighChol, Smoker, PhysActivity,
            HvyAlcoholConsump, GenHlth, Age, Sex
    Returns: (total_points, category, breakdown_list)
    """
    breakdown = []

    pts, label = _bmi_points(inputs["BMI"])
    breakdown.append((label, pts))

    pts_age, label_age = _age_points(inputs["Age"])
    breakdown.append((label_age, pts_age))

    if inputs["HighBP"] == 1:
        breakdown.append(("High blood pressure", 2))
    if inputs["HighChol"] == 1:
        breakdown.append(("High cholesterol", 1))
    if inputs["Smoker"] == 1:
        breakdown.append(("Smoker", 1))
    if inputs["PhysActivity"] == 0:
        breakdown.append(("No physical activity in past 30 days", 1))
    if inputs["HvyAlcoholConsump"] == 1:
        breakdown.append(("Heavy alcohol consumption", 1))
    if inputs["GenHlth"] >= 4:
        breakdown.append(("Self-rated general health poor/fair", 2))

    total = sum(p for _, p in breakdown)

    if total <= 2:
        category = "Lower risk"
    elif total <= 5:
        category = "Moderate risk"
    else:
        category = "Higher risk"

    return total, category, breakdown


# Max possible points, used to scale the score into a rough 0-1 for comparison
# with the ML probability (clearly an approximation, not a probability).
MAX_POINTS = 3 + 3 + 2 + 1 + 1 + 1 + 1 + 2  # = 14
