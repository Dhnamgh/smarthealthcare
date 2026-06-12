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

# ✅ TRANSFORMER
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="CardioAI", layout="wide")

# ================= LOAD =================
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load_model("models/heart_model.pkl")
stroke_model = load_model("models/stroke_model.pkl")

# ================= FEATURE SET =================
HEART = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
STROKE = ["age","gender","hypertension","heart_disease","ever_married","work_type","residence","avg_glucose","bmi","smoking"]

# ================= LOAD EMBEDDING MODEL =================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# ================= SEMANTIC MAPPING =================
def semantic_mapping_dl(df, expected_cols, threshold=0.6):

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    col_embeddings = embed_model.encode(list(df.columns))
    exp_embeddings = embed_model.encode(expected_cols)

    mapping = {}
    unmatched = []

    for i, col in enumerate(df.columns):

        sims = cosine_similarity([col_embeddings[i]], exp_embeddings)[0]

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score >= threshold:
            mapping[col] = expected_cols[best_idx]
        else:
            unmatched.append(col)

    df = df.rename(columns=mapping)

    return df, mapping, unmatched

# ================= PREPROCESS =================
def preprocess(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())

    return df

# ================= SCALING =================
def scale(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# ================= OUTLIER =================
def detect_outliers(df):
    z = np.abs((df - df.mean()) / df.std())
    return (z > 3).sum().sum()

# ================= SHAP =================
def show_shap(model, data, features):

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

# ================= UI =================
st.title("🧠💓 CardioAI Deep Learning Dashboard")

menu = st.sidebar.radio("Menu", ["Upload Dataset","Model Analysis"])

# ================= TEMPLATE =================
st.sidebar.header("📥 Templates")

def download(path):
    if os.path.exists(path):
        with open(path,"rb") as f:
            st.sidebar.download_button(path, f)

download("data/heart_template.csv")
download("data/stroke_template.csv")

# ================= FILE LOAD =================
def safe_read(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file, engine="openpyxl")
        else:
            st.error("Unsupported file format")
            st.stop()
    except Exception as e:
        st.error(f"File error: {e}")
        st.stop()

# ================= MAIN =================
if menu == "Upload Dataset":

    file = st.file_uploader("Upload CSV/Excel", ["csv","xlsx"])

    if file:

        df = safe_read(file)
        st.write("Preview:", df.head())

        model_choice = st.selectbox("Model", ["Heart","Stroke"])

        if model_choice == "Heart":
            df, mapping, unmatched = semantic_mapping_dl(df, HEART)
            cols = HEART
            model = heart_model
        else:
            df, mapping, unmatched = semantic_mapping_dl(df, STROKE)
            cols = STROKE
            model = stroke_model

        st.write("Mapping:", mapping)

        if unmatched:
            st.warning(f"Unmapped: {unmatched}")

        missing = [c for c in cols if c not in df.columns]

        if missing:
            st.error(f"Missing: {missing}")
            st.stop()

        df = preprocess(df)
        df = df[cols]

        st.warning(f"Outliers: {detect_outliers(df)}")

        df_scaled = scale(df)

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

    try:
        X_test, y_test = joblib.load("models/heart_test.pkl")

        metrics = evaluate(heart_model, X_test, y_test)

        df_metrics = pd.DataFrame(metrics.items(), columns=["Metric","Value"])

        st.table(df_metrics)

        df_metrics.to_csv("metrics.csv", index=False)

        st.download_button("Download Metrics", open("metrics.csv","rb"), "metrics.csv")

    except:
        st.warning("No test data")
