import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(page_title="CardioAI", layout="wide")

# ================= LOAD MODEL =================
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

heart_model = load_model("models/heart_model.pkl")
stroke_model = load_model("models/stroke_model.pkl")

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= UI HEADER =================
st.title("🧠💓 CardioAI - Smart Healthcare Dashboard")
st.markdown("Predict Heart Disease and Stroke Risk with Explainable Models")

# ================= SIDEBAR =================
menu = st.sidebar.radio(
    "📌 Menu",
    ["Prediction", "Upload Dataset", "Model Analysis", "History"]
)

st.sidebar.markdown("### 📥 Dataset")
st.sidebar.markdown("""
[Heart Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)  
[Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
""")

# ================= UTILS =================
def get_risk(prob):
    if prob < 0.3:
        return "LOW ✅"
    elif prob < 0.7:
        return "MEDIUM ⚠️"
    return "HIGH 🚨"


# ================= AUTO DETECT =================
def preprocess_df(df):
    df = df.copy()

    # convert categorical automatically
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # fill missing
    df = df.fillna(df.mean(numeric_only=True))

    return df


# ================= SHAP =================
def show_shap(model, data, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        st.subheader("🔍 Feature Impact")

        df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_values[1][0]
        }).sort_values(by="Impact", key=abs, ascending=False)

        st.bar_chart(df.set_index("Feature"))

        # ===== SHAP FORCE PLOT =====
        st.subheader("⚡ SHAP Force Plot")

        fig = plt.figure()
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][0],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig)

        # ===== SAVE FIG =====
        fig.savefig("shap_plot.png")

        st.download_button(
            "📥 Download SHAP figure",
            open("shap_plot.png", "rb"),
            file_name="shap_plot.png"
        )

    except Exception as e:
        st.warning(f"SHAP error: {e}")


# ================= METRICS =================
def show_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    # CONFUSION MATRIX
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        fig.savefig("confusion_matrix.png")
        st.download_button(
            "Download CM",
            open("confusion_matrix.png", "rb"),
            file_name="cm.png"
        )

    # ROC
    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.legend()
        st.pyplot(fig)

        fig.savefig("roc_curve.png")
        st.download_button(
            "Download ROC",
            open("roc_curve.png", "rb"),
            file_name="roc.png"
        )


# ================= 1. PREDICTION =================
if menu == "Prediction":

    tab1, tab2 = st.tabs(["❤️ Heart", "🧠 Stroke"])

    # HEART
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 20, 80, 40)
            cp = st.slider("Chest Pain", 0, 3)
            trestbps = st.slider("Blood Pressure", 80, 200, 120)

        with col2:
            chol = st.slider("Cholesterol", 100, 400, 200)
            thalach = st.slider("Heart Rate", 70, 210, 150)
            sex = st.selectbox("Sex", [0, 1])

        if st.button("Predict Heart"):
            data = np.array([[age, sex, cp, trestbps, chol,
                              0, 0, thalach, 0,
                              0, 1, 0, 2]])

            prob = heart_model.predict_proba(data)[0][1]

            st.metric("Risk", get_risk(prob))
            st.progress(float(prob))

            show_shap(
                heart_model,
                data,
                ["age","sex","cp","bp","chol","fbs","ecg",
                 "hr","exang","oldpeak","slope","ca","thal"]
            )

    # STROKE
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 1, 100, 40)
            hypertension = st.selectbox("Hypertension", [0, 1])
            heart_disease = st.selectbox("Heart Disease", [0, 1])

        with col2:
            glucose = st.slider("Glucose", 50, 300, 100)
            bmi = st.slider("BMI", 10.0, 50.0, 25.0)

        if st.button("Predict Stroke"):
            data = np.array([[age, hypertension, heart_disease,
                              1, 2, 0, glucose, bmi, 1]])

            prob = stroke_model.predict_proba(data)[0][1]

            st.metric("Risk", get_risk(prob))
            st.progress(float(prob))

            show_shap(
                stroke_model,
                data,
                ["age","htn","heart","married",
                 "work","res","glucose","bmi","smoke"]
            )


# ================= 2. UPLOAD =================
elif menu == "Upload Dataset":

    st.subheader("📂 Upload CSV")

    file = st.file_uploader("Upload dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        df = preprocess_df(df)

        model_option = st.selectbox("Model", ["Heart", "Stroke"])

        if st.button("Predict Dataset"):
            try:
                model = heart_model if model_option == "Heart" else stroke_model

                preds = model.predict(df)
                probs = model.predict_proba(df)[:, 1]

                df["Prediction"] = preds
                df["Probability"] = probs

                st.write(df)

                st.download_button(
                    "Download Result",
                    df.to_csv(index=False),
                    "result.csv"
                )

            except Exception as e:
                st.error(f"Data error: {e}")


# ================= 3. HISTORY =================
elif menu == "History":

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        st.line_chart(df["prob"])
    else:
        st.info("No history yet")

# ================= FOOTER =================
st.markdown("---")
st.caption("CardioAI - Research Dashboard")
