import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_curve, auc
from datetime import datetime

# ================= LOAD =================
heart_model = joblib.load("models/heart_model.pkl")
stroke_model = joblib.load("models/stroke_model.pkl")

heart_test = joblib.load("models/heart_test.pkl")
stroke_test = joblib.load("models/stroke_test.pkl")

# lưu lịch sử tạm (runtime)
if "history" not in st.session_state:
    st.session_state.history = []

# ================= CONFIG =================
st.set_page_config(page_title="Smart Healthcare AI", layout="wide")

st.title("🧠💓 SMART HEALTHCARE AI DASHBOARD")

# ================= UTILS =================
def get_risk(prob):
    if prob < 0.3:
        return "LOW ✅"
    elif prob < 0.7:
        return "MEDIUM ⚠️"
    else:
        return "HIGH 🚨"

def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

def show_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    st.pyplot(fig)

def show_shap(model, input_data, feature_names):
    st.subheader("🔍 SHAP Explainability")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    values = shap_values[1][0]

    df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": values
    }).sort_values(by="Impact", key=abs, ascending=False)

    st.write(df)
    st.bar_chart(df.set_index("Feature"))

# ================= SIDEBAR =================
menu = st.sidebar.radio(
    "📌 Menu",
    ["Prediction", "Model Analysis", "Batch Prediction", "History"]
)

# ================= PREDICTION =================
if menu == "Prediction":

    option = st.selectbox("Chọn chức năng", ["Heart Disease", "Stroke"])

    col1, col2 = st.columns(2)

    # ========= HEART =========
    if option == "Heart Disease":
        st.subheader("❤️ Heart Disease Prediction")

        with col1:
            age = st.slider("Age", 20, 80, 40)
            cp = st.slider("Chest Pain", 0, 3)
            trestbps = st.slider("Blood Pressure", 80, 200, 120)

        with col2:
            chol = st.slider("Cholesterol", 100, 400, 200)
            thalach = st.slider("Heart Rate", 70, 210, 150)
            sex = st.selectbox("Sex (0=female,1=male)", [0, 1])

        if st.button("Predict Heart"):

            data = np.array([[age, sex, cp, trestbps, chol,
                              0, 0, thalach, 0,
                              0, 1, 0, 2]])

            proba = heart_model.predict_proba(data)[0][1]
            pred = heart_model.predict(data)[0]

            st.metric("Risk", get_risk(proba))
            st.progress(float(proba))

            # lưu history
            st.session_state.history.append({
                "time": datetime.now(),
                "type": "heart",
                "prob": proba
            })

            # SHAP
            show_shap(
                heart_model,
                data,
                ["age","sex","cp","bp","chol","fbs","ecg",
                 "hr","exang","oldpeak","slope","ca","thal"]
            )

    # ========= STROKE =========
    else:
        st.subheader("🧠 Stroke Prediction")

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

            proba = stroke_model.predict_proba(data)[0][1]
            pred = stroke_model.predict(data)[0]

            st.metric("Risk", get_risk(proba))
            st.progress(float(proba))

            st.session_state.history.append({
                "time": datetime.now(),
                "type": "stroke",
                "prob": proba
            })

            show_shap(
                stroke_model,
                data,
                ["age","hypertension","heart_disease",
                 "married","work","residence",
                 "glucose","bmi","smoking"]
            )

# ================= ANALYSIS =================
elif menu == "Model Analysis":

    option = st.selectbox("Chọn model", ["Heart", "Stroke"])

    if option == "Heart":
        X_test, y_test = heart_test
        model = heart_model
    else:
        X_test, y_test = stroke_test
        model = stroke_model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Confusion Matrix")
        show_confusion_matrix(y_test, y_pred)

    with col2:
        st.subheader("📈 ROC Curve")
        show_roc(y_test, y_prob)

# ================= BATCH =================
elif menu == "Batch Prediction":

    st.subheader("📂 Upload CSV")

    file = st.file_uploader("Upload file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        if st.button("Predict Batch"):
            preds = heart_model.predict(df)
            df["Prediction"] = preds

            st.write(df)
            st.download_button(
                "Download",
                df.to_csv(index=False),
                "result.csv"
            )

# ================= HISTORY =================
elif menu == "History":

    st.subheader("📜 Prediction History")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.plot(df["prob"])
        ax.set_title("Risk Trend")
        st.pyplot(fig)
    else:
        st.info("Chưa có dữ liệu")

# ================= FOOTER =================
st.markdown("---")
st.caption("Smart Healthcare AI - Student Research Project")
