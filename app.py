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

# ================= SEMANTIC DICT =================
SEMANTIC_DICT = {
    "age": ["age","patient_age","years"],
    "sex": ["sex","gender"],
    "cp": ["cp","chest_pain"],
    "trestbps": ["bp","blood_pressure"],
    "chol": ["chol","cholesterol"],
    "thalach": ["heart_rate","pulse","max_hr"],
    "exang": ["angina"],
    "oldpeak": ["st_depression"],
    "slope": ["slope"],
    "ca": ["vessels"],
    "thal": ["thal"],

    "avg_glucose": ["glucose","glucose_level","sugar"],
    "bmi": ["bmi","body_mass"],
    "smoking": ["smoking","smoker"],
    "hypertension": ["hypertension","high_bp"],
    "heart_disease": ["heart_disease","cardiac"],
    "ever_married": ["married"],
    "work_type": ["job","occupation"],
    "residence": ["residence","location"]
}

# ================= HYBRID MAP =================
def hybrid_map(df, features):
    df = df.copy()
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip().str.lower()

    mapping = {}
    unmatched = []

    for col in df.columns:
        mapped = None

        if col in features:
            mapped = col

        if not mapped:
            for key, vals in SEMANTIC_DICT.items():
                for v in vals:
                    if v in col:
                        mapped = key
                        break
                if mapped:
                    break

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
    df = df.fillna(df.mean(numeric_only=True))
    return df

# ================= SCALE =================
def scale(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# ================= OUTLIER =================
def detect_outliers(df):
    z = np.abs((df - df.mean()) / df.std())
    return (z > 3).sum().sum()

# ================= VALIDATION =================
def validate_medical(df, model_type):

    issues = []

    def check_range(col, low, high):
        if col in df.columns:
            bad = df[(df[col] < low) | (df[col] > high)]
            if len(bad) > 0:
                issues.append(f"{col}: {len(bad)} values outside [{low}, {high}]")

    if model_type == "Heart":
        check_range("age", 20, 100)
        check_range("trestbps", 80, 200)
        check_range("chol", 120, 400)
        check_range("thalach", 60, 210)
        check_range("oldpeak", 0, 6)

    else:
        check_range("age", 0, 120)
        check_range("avg_glucose", 50, 300)
        check_range("bmi", 10, 60)

    return issues

# ================= SHAP =================
def show_shap(model, data, features):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        df = pd.DataFrame({
            "Feature": features,
            "Impact": shap_values[1][0]
        }).sort_values(by="Impact", key=abs, ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(df.set_index("Feature"))

    except:
        st.warning("SHAP not supported")

# ================= SAFE FILE =================
def safe_read(file):
    if file is None:
        return None

    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(file, engine="openpyxl")
        else:
            st.error("Only CSV/Excel supported")
            return None

        df.columns = df.columns.astype(str)
        df.columns = df.columns.str.strip().str.lower()

        if df.shape[0] == 0:
            st.error("Empty file")
            return None

        return df

    except Exception as e:
        st.error(str(e))
        return None

# ================= UI =================
st.title("🧠 CardioAI Healthcare System")

menu = st.sidebar.radio("Menu", ["Upload Dataset","Model Analysis"])

# ================= MAIN =================
if menu == "Upload Dataset":

    file = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
    df = safe_read(file)

    if df is not None:

        st.write(df.head())

        model_choice = st.selectbox("Model", ["Heart","Stroke"])

        if model_choice == "Heart":
            df, mapping, unmatched = hybrid_map(df, HEART)
            cols = HEART
            model = heart_model
        else:
            df, mapping, unmatched = hybrid_map(df, STROKE)
            cols = STROKE
            model = stroke_model

        st.write("Mapping:", mapping)

        if unmatched:
            st.warning(f"Unmapped: {unmatched}")

        missing = [c for c in cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # ✅ VALIDATION
        issues = validate_medical(df, model_choice)
        if len(issues) > 0:
            st.error("⚠️ Medical Data Issues:")
            for i in issues:
                st.write("-", i)
        else:
            st.success("✅ Data looks valid")

        df_proc = preprocess(df)[cols]

        outliers = detect_outliers(df_proc)
        st.warning(f"Outliers detected: {outliers}")

        df_scaled = scale(df_proc)

        if st.button("Predict"):

            preds = model.predict(df_scaled)
            probs = model.predict_proba(df_scaled)[:,1]

            df["Prediction"] = preds
            df["Probability"] = probs

            st.write(df)

            st.download_button("Download Result", df.to_csv(index=False), "result.csv")

            show_shap(model, df_scaled.values[:1], cols)

elif menu == "Model Analysis":

    try:
        X_test, y_test = joblib.load("models/heart_test.pkl")
        preds = heart_model.predict(X_test)
        prob = heart_model.predict_proba(X_test)[:,1]

        st.write({
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "AUC": auc(*roc_curve(y_test, prob)[:2])
        })

    except:
        st.warning("No test data")
