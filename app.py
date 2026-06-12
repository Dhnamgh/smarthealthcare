import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
st.set_page_config(page_title="CardioAI Healthcare", layout="wide")

# ================= LOAD MODEL =================
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load_model("models/heart_model.pkl")
stroke_model = load_model("models/stroke_model.pkl")

# ================= FEATURES =================
HEART = ["age","sex","cp","trestbps","chol","fbs","restecg",
         "thalach","exang","oldpeak","slope","ca","thal"]

STROKE = ["age","gender","hypertension","heart_disease",
          "ever_married","work_type","residence",
          "avg_glucose","bmi","smoking"]

# ================= CLEAN DATA =================
def preprocess(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("category").cat.codes

    df = df.fillna(df.mean(numeric_only=True))
    return df

def scale(df):
    return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

# ================= SAFE READ =================
def safe_read(file):
    if file is None:
        return None

    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine="openpyxl")

        df.columns = df.columns.str.strip().str.lower()
        return df

    except:
        st.error("File error")
        return None

# ================= UI HEADER =================
st.markdown("""
<h2 style='color:#1f77b4'>🏥 CardioAI Smart Healthcare System</h2>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
menu = st.tabs([
    "🏠 Trang chủ",
    "📊 Dự đoán",
    "📂 Upload dữ liệu",
    "📈 Dashboard",
    "💬 Hỏi đáp"
])

# ================= HOME =================
with menu[0]:
    st.markdown("""
### Hệ thống hỗ trợ chẩn đoán bệnh tim và đột quỵ

- ✔ Dự đoán nguy cơ bệnh
- ✔ Phân tích dữ liệu y khoa
- ✔ Hỗ trợ bác sĩ và bệnh nhân
""")

# ================= PREDICT =================
with menu[1]:

    st.subheader("Dự đoán nhanh")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 80, 40)
        chol = st.slider("Cholesterol", 100, 400, 200)

    with col2:
        bp = st.slider("Blood Pressure", 80, 200, 120)
        hr = st.slider("Heart Rate", 60, 200, 150)

    if st.button("Predict Heart Risk"):

        data = np.array([[age,1,2,bp,chol,0,1,hr,0,1.2,1,0,2]])
        prob = heart_model.predict_proba(data)[0][1]

        st.metric("Risk Score", f"{prob:.2f}")

# ================= UPLOAD =================
with menu[2]:

    st.subheader("Upload dữ liệu bệnh nhân")

    file = st.file_uploader("Chọn file CSV hoặc Excel")

    df = safe_read(file)

    if df is not None:

        st.success("Đã tải dữ liệu")

        model_choice = st.selectbox("Chọn mô hình", ["Heart","Stroke"])

        if st.button("Chạy dự đoán"):

            df = preprocess(df)

            if model_choice == "Heart":
                df = df[HEART]
                model = heart_model
            else:
                df = df[STROKE]
                model = stroke_model

            df_scaled = scale(df)

            preds = model.predict(df_scaled)
            probs = model.predict_proba(df_scaled)[:,1]

            result = df.copy()
            result["Prediction"] = preds
            result["Probability"] = probs

            st.success("Hoàn thành dự đoán")

            st.dataframe(result)

            st.download_button(
                "📥 Tải kết quả",
                result.to_csv(index=False),
                "result.csv"
            )

# ================= DASHBOARD =================
with menu[3]:

    st.subheader("Dashboard")

    st.info("Chức năng hiển thị biểu đồ sẽ đặt ở đây")

# ================= CHAT =================
with menu[4]:

    st.subheader("💬 Trợ lý y tế")

    question = st.text_input("Nhập câu hỏi")

    if question:
        if "tim" in question.lower():
            st.write("Bệnh tim liên quan đến huyết áp, cholesterol và lối sống.")
        elif "đột quỵ" in question:
            st.write("Đột quỵ liên quan đến huyết áp cao và đường huyết.")
        else:
            st.write("Vui lòng hỏi về bệnh tim hoặc đột quỵ.")
