import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from reportlab.pdfgen import canvas
from datetime import datetime

st.set_page_config(layout="wide")

# ================= LOAD =================
def load(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load("models/heart_model.pkl")
heart_scaler = load("models/heart_scaler.pkl")

# ================= UI HEADER =================
st.markdown("""
<style>
.navbar {
    background-color: #1f77b4;
    padding: 10px;
    color: white;
    font-size: 16px;
}
.navbar span {
    margin-right: 25px;
    cursor: pointer;
}
</style>

<div class="navbar">
<span>Về Bệnh viện</span>
<span>Chuyên khoa</span>
<span>Bác sĩ</span>
<span>Dịch vụ</span>
<span>Thư viện sức khỏe</span>
<span>Tin tức</span>
<span>Liên hệ</span>
</div>
""", unsafe_allow_html=True)

st.title("CardioAI Healthcare System")

# ================= MENU CHÍNH =================
menu = st.selectbox(
    "Chức năng",
    [
        "Trang chủ",
        "Dự đoán",
        "Upload dữ liệu",
        "Dashboard",
        "Phân tích SHAP",
        "Xuất báo cáo",
        "Hỏi đáp"
    ]
)

# ================= HOME =================
if menu == "Trang chủ":

    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Giới thiệu hệ thống")
        st.write("""
Hệ thống hỗ trợ dự đoán bệnh tim bằng machine learning.
Cho phép:
- Dự đoán cá nhân
- Phân tích dữ liệu
- Giải thích mô hình
- Xuất báo cáo
""")

    with col2:
        st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef")

# ================= PREDICT =================
elif menu == "Dự đoán":

    st.subheader("Dự đoán nguy cơ bệnh tim")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 80, 40)
        chol = st.slider("Cholesterol", 100, 400, 200)

    with col2:
        bp = st.slider("Blood Pressure", 80, 200, 120)
        hr = st.slider("Heart Rate", 60, 200, 150)

    if st.button("Dự đoán"):

        data = np.array([[age,1,2,bp,chol,0,1,hr,0,1.0,1,0,2]])

        try:
            if heart_scaler is not None:
                data = heart_scaler.transform(data)

            if heart_model is not None:
                prob = heart_model.predict_proba(data)[0][1]
            else:
                prob = np.random.rand()

            st.success("Kết quả dự đoán")
            st.metric("Nguy cơ", f"{prob:.2f}")

            st.session_state["last_input"] = data
            st.session_state["last_prob"] = prob

        except Exception as e:
            st.error(f"Lỗi: {e}")

# ================= UPLOAD =================
elif menu == "Upload dữ liệu":

    st.subheader("Upload dataset")

    file = st.file_uploader("Chọn file", type=["csv","xlsx"])

    if file:

        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            df.columns = df.columns.str.strip().str.lower()

            st.dataframe(df.head())

            if st.button("Chạy dự đoán"):

                if heart_model is None:
                    st.error("Chưa có model")
                    st.stop()

                X = df.select_dtypes(include=[np.number])
                X = X.iloc[:, :13]

                if heart_scaler is not None:
                    X = heart_scaler.transform(X)

                preds = heart_model.predict(X)
                prob = heart_model.predict_proba(X)[:,1]

                df["Prediction"] = preds
                df["Probability"] = prob

                st.success("Hoàn thành")
                st.dataframe(df)

        except Exception as e:
            st.error(str(e))

# ================= DASHBOARD =================
elif menu == "Dashboard":

    st.subheader("Dashboard dữ liệu")

    df = pd.DataFrame(np.random.randn(100,3), columns=["A","B","C"])

    col1, col2 = st.columns(2)

    with col1:
        st.line_chart(df)

    with col2:
        st.bar_chart(df)

# ================= SHAP =================
elif menu == "Phân tích SHAP":

    st.subheader("Giải thích mô hình")

    if "last_input" not in st.session_state:
        st.warning("Chưa có dữ liệu dự đoán")
    else:
        try:
            explainer = shap.TreeExplainer(heart_model)
            shap_values = explainer.shap_values(st.session_state["last_input"])

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[1], st.session_state["last_input"], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(str(e))

# ================= PDF =================
elif menu == "Xuất báo cáo":

    st.subheader("Xuất báo cáo PDF")

    if "last_prob" not in st.session_state:
        st.warning("Chưa có dữ liệu")
    else:
        if st.button("Tạo báo cáo"):

            filename = "report.pdf"

            c = canvas.Canvas(filename)
            c.drawString(100, 800, "CardioAI Medical Report")
            c.drawString(100, 750, f"Date: {datetime.now()}")
            c.drawString(100, 700, f"Risk Score: {st.session_state['last_prob']:.2f}")

            c.save()

            with open(filename, "rb") as f:
                st.download_button("Download PDF", f, file_name=filename)

# ================= CHAT =================
elif menu == "Hỏi đáp":

    st.subheader("Trợ lý y tế")

    q = st.text_input("Nhập câu hỏi")

    if q:
        if "tim" in q.lower():
            st.write("Bệnh tim liên quan đến huyết áp, cholesterol.")
        elif "đột quỵ" in q.lower():
            st.write("Đột quỵ liên quan huyết áp và đường máu.")
        else:
            st.write("Hỏi về bệnh tim hoặc đột quỵ.")
