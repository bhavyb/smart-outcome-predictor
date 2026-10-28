import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎯", layout="wide")

# Paths are built from this file's location, so the app works from any working directory
BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA = BASE / "data" / "Smart_Outcome_Predictor_Dataset_5200.xlsx"

NUM_FEATURES = ["age", "week_of_year", "start_month", "sessions", "time_spent_hours", "videos_watched",
                "quiz_attempts", "assignments_submitted", "forum_posts", "avg_quiz_score", "attendance_rate"]
CAT_FEATURES = ["country_region", "device_type", "education_background", "course_level", "course_category"]


def build_meta(df):
    return {"num_features": NUM_FEATURES, "cat_features": CAT_FEATURES,
            "best_classifier": "Gradient Boosting Classifier", "best_regressor": "Gradient Boosting Regressor",
            "num_ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in NUM_FEATURES},
            "cat_options": {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_FEATURES}}


def train_models():
    """Fallback: retrain the two best models (same pipeline/params/split as the notebook)
    inside the current environment, so version mismatches of saved .joblib files never break the app."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_excel(DATA)
    df["start_month"] = pd.to_datetime(df["course_start_date"]).dt.month
    X = df[NUM_FEATURES + CAT_FEATURES]
    X_tr, _, yc_tr, _, yr_tr, _ = train_test_split(X, df["completion_status"], df["final_score"],
                                                   test_size=0.2, random_state=42, stratify=df["completion_status"])

    def pipe(model):
        prep = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_FEATURES),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES)])
        return Pipeline([("prep", prep), ("model", model)])

    clf = pipe(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)).fit(X_tr, yc_tr)
    reg = pipe(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)).fit(X_tr, yr_tr)
    return clf, reg, build_meta(df)


@st.cache_resource(show_spinner="Loading models...")
def load():
    try:
        clf = joblib.load(MODELS / "best_classifier.joblib")
        reg = joblib.load(MODELS / "best_regressor.joblib")
        meta = json.load(open(MODELS / "metadata.json"))
        return clf, reg, meta, False
    except Exception:
        # saved models missing or built with different library versions -> retrain here
        clf, reg, meta = train_models()
        return clf, reg, meta, True


clf, reg, meta, retrained = load()
if retrained:
    st.caption("ℹ️ Models were re-trained at startup in this environment (same pipeline and parameters as the notebook).")
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
    for title, f in [("Classification results (test set)", "classification_results.csv"),
                     ("Regression results (test set)", "regression_results.csv")]:
        st.subheader(title)
        if (MODELS / f).exists():
            st.dataframe(pd.read_csv(MODELS / f).round(4), use_container_width=True)
        else:
            st.info(f"{f} not found. Run the notebook to generate it.")