"""
app.py  -  Real-time Cardio-Metabolic Risk Checker
Loads the best model saved by the notebook and scores the user two ways, live.
"""

import json
import joblib
import pandas as pd
import streamlit as st

from rule_engine import score as rule_score, MAX_POINTS
from explain import explain_row

st.set_page_config(page_title="Cardio-Metabolic Risk Checker", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.joblib")
    background = joblib.load("shap_background.joblib")
    meta = json.load(open("model_meta.json"))
    return model, background, meta


model, background, meta = load_artifacts()
FEATURES = meta["features"]

st.title("Cardio-Metabolic Risk Checker")
st.caption(
    f"Research/educational prototype - NOT a diagnosis. ML model: "
    f"{meta['best_model']} (test AUC {meta['metrics']['AUC']}), trained on the "
    f"CDC BRFSS 2015 self-reported survey data. Target: pre-diabetes or diabetes."
)

st.sidebar.header("Your details")
bmi = st.sidebar.slider("BMI", 12.0, 60.0, 27.0, 0.1)
age = st.sidebar.select_slider(
    "Age group", options=list(range(1, 14)), value=7,
    format_func=lambda a: {1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",
                           6:"45-49",7:"50-54",8:"55-59",9:"60-64",10:"65-69",
                           11:"70-74",12:"75-79",13:"80+"}[a],
)
sex = st.sidebar.radio("Sex", [0, 1], format_func=lambda s: "Female" if s == 0 else "Male")
gen_hlth = st.sidebar.slider("General health (1 excellent - 5 poor)", 1, 5, 3)
high_bp = st.sidebar.checkbox("High blood pressure")
high_chol = st.sidebar.checkbox("High cholesterol")
smoker = st.sidebar.checkbox("Smoker (100+ cigarettes in life)")
phys = st.sidebar.checkbox("Physically active in past 30 days", value=True)
heavy_alc = st.sidebar.checkbox("Heavy alcohol consumption")

inputs = {"BMI": bmi, "HighBP": int(high_bp), "HighChol": int(high_chol),
          "Smoker": int(smoker), "PhysActivity": int(phys),
          "HvyAlcoholConsump": int(heavy_alc), "GenHlth": gen_hlth,
          "Age": age, "Sex": sex}
row = pd.DataFrame([inputs])[FEATURES]

c1, c2 = st.columns(2)

with c1:
    st.subheader("Rule-based scorer")
    total, category, breakdown = rule_score(inputs)
    st.metric("Risk category", category, f"{total} / {MAX_POINTS} points")
    st.write("**Why (every point is a rule):**")
    for label, pts in breakdown:
        st.write(f"- +{pts}  {label}" if pts > 0 else f"- {label}")

with c2:
    st.subheader(f"ML model ({meta['best_model']})")
    prob = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated probability", f"{prob*100:.0f}%")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption("Probability of pre-diabetes or diabetes for this profile.")

rule_frac = total / MAX_POINTS
st.divider()
if abs(rule_frac - prob) < 0.15:
    st.success(f"Methods broadly AGREE (rule {rule_frac*100:.0f}% vs ML {prob*100:.0f}%).")
else:
    st.warning(f"Methods DISAGREE (rule {rule_frac*100:.0f}% vs ML {prob*100:.0f}%). "
               "Worth discussing in your thesis: which do you trust, and why?")

st.divider()
if st.button("Explain the ML estimate (SHAP)"):
    contrib = explain_row(model, background, row, FEATURES)
    st.write("**What pushed this estimate up (+) or down (-) for this person:**")
    for feat, val in contrib:
        st.write(f"- **{feat}**: {val:+.3f}  ({'increases' if val > 0 else 'decreases'} risk)")
    st.caption("Compare this with the rule breakdown on the left - do the two "
               "methods point at the same drivers?")