import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎯", layout="wide")

@st.cache_resource
def load():
    clf = joblib.load("models/best_classifier.joblib")
    reg = joblib.load("models/best_regressor.joblib")
    meta = json.load(open("models/metadata.json"))
    return clf, reg, meta

clf, reg, meta = load()
INT_COLS = {"age", "week_of_year", "start_month", "sessions", "videos_watched",
            "quiz_attempts", "assignments_submitted", "forum_posts"}

st.title("🎯 Smart Outcome Predictor")
st.caption(f"Ensemble models  |  Classifier: **{meta['best_classifier']}**  |  Regressor: **{meta['best_regressor']}**")

tab1, tab2 = st.tabs(["🔮 Predict", "📊 Model Comparison"])

with tab1:
    st.subheader("Enter student details")
    inputs = {}
    cols = st.columns(3)
    for i, c in enumerate(meta["num_features"]):
        lo, hi, med = meta["num_ranges"][c]
        with cols[i % 3]:
            if c in INT_COLS:
                inputs[c] = st.number_input(c, int(lo), int(hi), int(med), step=1)
            else:
                inputs[c] = st.number_input(c, float(lo), float(hi), float(round(med, 2)), step=0.1)
    cols = st.columns(3)
    for i, c in enumerate(meta["cat_features"]):
        with cols[i % 3]:
            inputs[c] = st.selectbox(c, meta["cat_options"][c])

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([inputs])[meta["num_features"] + meta["cat_features"]]
        proba = float(clf.predict_proba(row)[0, 1])
        score = float(min(100, max(0, reg.predict(row)[0])))
        c1, c2 = st.columns(2)
        c1.metric("Course completion probability", f"{proba:.1%}")
        if proba >= 0.5:
            c1.success("Likely to COMPLETE ✅")
        else:
            c1.error("Likely NOT to complete ❌")
        c2.metric("Predicted final score", f"{score:.1f} / 100")
        c2.progress(int(score))

with tab2:
    st.subheader("Classification results (test set)")
    st.dataframe(pd.read_csv("models/classification_results.csv").round(4), use_container_width=True)
    st.subheader("Regression results (test set)")
    st.dataframe(pd.read_csv("models/regression_results.csv").round(4), use_container_width=True)