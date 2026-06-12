import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

from difflib import get_close_matches

st.set_page_config(page_title="CardioAI", layout="wide")

# ================= LOAD =================
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load_model("models/heart_model.pkl")
stroke_model = load_model("models/stroke_model.pkl")

# ================= FEATURE SET =================
HEART = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
STROKE = ["age","gender","hypertension","heart_disease","ever_married","work_type","residence","avg_glucose","bmi","smoking"]

# ================= SEMANTIC DICTIONARY =================
SEMANTIC_DICT = {
    "age": ["age","patient_age","years"],
    "sex": ["sex","gender"],
    "cp": ["cp","chest_pain"],
    "trestbps": ["bp","blood_pressure"],
    "chol": ["chol","cholesterol"],
    "thalach": ["heart_rate","pulse","max_hr","heart_rate_max"],
    "exang": ["angina","exercise_angina"],
    "oldpeak": ["st_depression"],
    "slope": ["slope"],
    "ca": ["vessels"],
    "thal": ["thal"],

    "avg_glucose": ["glucose","glucose_level","sugar"],
    "bmi": ["bmi","body_mass","weight_index"],
    "smoking": ["smoking","smoker"],
    "hypertension": ["hypertension","high_bp"],
    "heart_disease": ["heart_disease","cardiac"],
    "ever_married": ["married"],
    "work_type": ["job","occupation"],
    "residence": ["residence","location"]
}

# ================= HYBRID MAPPING =================
def hybrid_map(df, features):

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    mapping = {}
    unmatched = []

    for col in df.columns:

        mapped = None

        # 1. exact
        if col in features:
            mapped = col

        # 2. semantic
        if not mapped:
            for key, vals in SEMANTIC_DICT.items():
                for v in vals:
                    if v in col:
                        mapped = key
                        break
                if mapped:
                    break

        # 3. fuzzy fallback
        if not mapped:
            match = get_close_matches(col, features, n=1, cutoff=0.6)
            if match:
                mapped = match[0]

        if mapped:
            mapping[col] = mapped
        else:
            unmatched.append(col)

    return df.rename(columns=mapping), mapping, unmatched

# ================= PREPROCESS =================
def preprocess(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())

    return df

# ================= SCALE =================
def scale(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# ================= OUTLIER =================
def detect_outliers(df):
    z = np.abs((df - df.mean()) / df.std())
    return (z > 3).sum().sum()

# ================= SHAP =================
def show_shap(model, data, features):

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        df = pd.DataFrame({
            "Feature": features,
            "Impact": shap_values[1][0]
        }).sort_values(by="Impact", key=abs, ascending=False)

        st.subheader("🔍 Feature Importance")
        st.bar_chart(df.set_index("Feature"))

        fig = plt.figure()
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], matplotlib=True, show=False)
        st.pyplot(fig)

        fig.savefig("shap.png")

        with open("shap.png", "rb") as f:
            st.download_button("Download SHAP", f, "shap.png")

    except:
        st.warning("SHAP not supported by this model")

# ================= METRICS =================
def evaluate(model, X, y):

    pred = model.predict(X)
    prob = model.predict_proba(X)[:,1]

    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "AUC": auc(*roc_curve(y, prob)[:2])
    }

# ================= SAFE FILE LOAD =================
def safe_read(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
