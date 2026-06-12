import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="CardioAI", layout="wide")

# ================= LOAD =================
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load_model("models/heart_model.pkl")
stroke_model = load_model("models/stroke_model.pkl")

# ================= FEATURES =================
HEART = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
STROKE = ["age","gender","hypertension","heart_disease","ever_married","work_type","residence","avg_glucose","bmi","smoking"]

# ================= AUTO MAP =================
def auto_map(df, features):
    mapping = {}
    for col in df.columns:
        for f in features:
            if f in col.lower() or col.lower() in f:
                mapping[col] = f
    return df.rename(columns=mapping)

# ================= PREPROCESS =================
def preprocess(df):
    df = df.copy()

    # categorical → numeric
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # xử lý missing
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())

    return df

# ================= SCALING =================
def scale_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)

# ================= OUTLIER =================
def detect_outliers(df):
    z = np.abs((df - df.mean()) / df.std())
    outliers = (z > 3).sum().sum()
    return outliers

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

# ================= SHAP =================
def show_shap(model, data, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    df = pd.DataFrame({
        "Feature": features,
        "Impact": shap_values[1][0]
    }).sort_values(by="Impact", key=abs, ascending=False)

    st.subheader("Feature Importance")
    st.bar_chart(df.set_index("Feature"))

    fig = plt.figure()
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], matplotlib=True, show=False)
    st.pyplot(fig)

    fig.savefig("shap.png")
    st.download_button("Download SHAP", open("shap.png","rb"), "shap.png")

# ================= UI =================
st.title("🧠 CardioAI Research Dashboard")

menu = st.sidebar.radio(
    "Menu",
    ["Upload Dataset","Model Analysis"]
)

# ================= TEMPLATE =================
st.sidebar.header("Templates")

def download(path):
    if os.path.exists(path):
        with open(path,"rb") as f:
            st.sidebar.download_button(path, f)

download("data/heart_template.csv")
download("data/stroke_template.csv")

# ================= UPLOAD =================
if menu == "Upload Dataset":

    file = st.file_uploader("Upload CSV/Excel", ["csv","xlsx"])

    if file:

        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.write(df.head())

        model_choice = st.selectbox("Model", ["Heart","Stroke"])

        if model_choice == "Heart":
            df = auto_map(df, HEART)
            cols = HEART
            model = heart_model
        else:
            df = auto_map(df, STROKE)
            cols = STROKE
            model = stroke_model

        missing = [c for c in cols if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        df = preprocess(df)
        df = df[cols]

        # OUTLIER
        outliers = detect_outliers(df)
        st.warning(f"Outliers detected: {outliers}")

        # SCALING
        df_scaled = scale_data(df)

        if st.button("Predict"):

            preds = model.predict(df_scaled)
            probs = model.predict_proba(df_scaled)[:,1]

            df["Prediction"] = preds
            df["Probability"] = probs

            st.write(df)

            st.download_button("Download Result", df.to_csv(index=False), "result.csv")

            show_shap(model, df_scaled.values[:1], df.columns)

# ================= ANALYSIS =================
elif menu == "Model Analysis":

    st.subheader("Model Performance")

    try:
        X_test, y_test = joblib.load("models/heart_test.pkl")

        metrics = evaluate(heart_model, X_test, y_test)

        df_metrics = pd.DataFrame(metrics.items(), columns=["Metric","Value"])
        st.table(df_metrics)

        # export table
        df_metrics.to_csv("metrics.csv", index=False)

        st.download_button(
            "Download Metrics",
            open("metrics.csv","rb"),
            "metrics.csv"
        )

    except:
        st.warning("No test data found")

# ================= INFO =================
st.info("""
✅ Auto column mapping
✅ Auto preprocessing
✅ Auto scaling
✅ Outlier detection
✅ SHAP explainability
✅ Export results for research
""")
