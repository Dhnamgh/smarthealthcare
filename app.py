import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

# ================= LOAD MODEL =================
def load(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load("models/heart_model.pkl")
heart_scaler = load("models/heart_scaler.pkl")

# ================= STYLE =================
st.markdown("""
<style>
.header {
    background-color: #1f5fa7;
    padding: 15px;
    color: white;
}
.menu {
    background-color: #ffffff;
    padding: 10px;
    border-bottom: 1px solid #ddd;
}
.menu span {
    margin-right: 30px;
    cursor: pointer;
    font-weight: 500;
}
.card {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
}
</style>

<div class="header">
<h2>BỆNH VIỆN ĐẠI HỌC Y DƯỢC TP.HCM</h2>
</div>

<div class="menu">
<span>Về Bệnh viện</span>
<span>Chuyên khoa</span>
<span>Bác sĩ</span>
<span>Dịch vụ</span>
<span>Thư viện sức khỏe</span>
<span>Tin tức & Sự kiện</span>
<span>Hỗ trợ người bệnh</span>
</div>
""", unsafe_allow_html=True)

# ================= MENU =================
menu = st.selectbox(
    "",
    [
        "Trang chủ",
        "Dự đoán",
        "Upload dữ liệu",
        "Dashboard",
        "Giải thích mô hình",
        "Báo cáo",
        "Hỏi đáp"
    ]
)

# ================= HOME =================
if menu == "Trang chủ":

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Giới thiệu")
        st.write("""
Hệ thống hỗ trợ dự đoán bệnh tim dựa trên dữ liệu y khoa.
Ứng dụng trong sàng lọc và hỗ trợ quyết định lâm sàng.
""")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1588776814546-ec7e5ef88b14")

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

    if st.button("Chạy dự đoán"):

        try:
            data = np.array([[age,1,2,bp,chol,0,1,hr,0,1.0,1,0,2]])

            if heart_scaler is not None:
                data = heart_scaler.transform(data)

            if heart_model is not None:
                prob = heart_model.predict_proba(data)[0][1]
            else:
                prob = (age/80 + chol/400 + bp/200) / 3

            st.success("Kết quả")
            st.metric("Nguy cơ", f"{prob:.2f}")

            st.session_state["data"] = data
            st.session_state["prob"] = prob

        except:
            st.error("Lỗi dữ liệu")

# ================= UPLOAD =================
elif menu == "Upload dữ liệu":

    st.subheader("Upload dữ liệu")

    file = st.file_uploader("Chọn file", type=["csv","xlsx"])

    if file:

        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            df.columns = df.columns.str.strip().str.lower()

            st.dataframe(df.head())

            if st.button("Dự đoán batch"):

                X = df.select_dtypes(include=np.number).iloc[:, :13]

                if heart_scaler is not None:
                    X = heart_scaler.transform(X)

                if heart_model is not None:
                    df["Prediction"] = heart_model.predict(X)
                    df["Probability"] = heart_model.predict_proba(X)[:,1]
                else:
                    df["Prediction"] = np.random.randint(0,2,len(df))
                    df["Probability"] = np.random.rand(len(df))

                st.success("Hoàn thành")
                st.dataframe(df)

                st.download_button(
                    "Tải kết quả",
                    df.to_csv(index=False),
                    "result.csv"
                )

        except:
            st.error("File lỗi")

# ================= DASHBOARD =================
elif menu == "Dashboard":

    st.subheader("Dashboard")

    data = pd.DataFrame(np.random.randn(50,3), columns=["A","B","C"])

    col1, col2 = st.columns(2)

    with col1:
        st.line_chart(data)

    with col2:
        st.area_chart(data)

# ================= SHAP =================
elif menu == "Giải thích mô hình":

    st.subheader("Giải thích")

    if "data" not in st.session_state:
        st.warning("Chưa có dữ liệu")
    else:
        try:
            import shap
            explainer = shap.TreeExplainer(heart_model)
            shap_values = explainer.shap_values(st.session_state["data"])

            fig = plt.figure()
            shap.summary_plot(shap_values[1], st.session_state["data"], show=False)
            st.pyplot(fig)
        except:
            st.warning("Không hỗ trợ SHAP")

# ================= REPORT =================
elif menu == "Báo cáo":

    st.subheader("Xuất báo cáo")

    if "prob" not in st.session_state:
        st.warning("Chưa có dữ liệu")
    else:
        if st.button("Tạo báo cáo"):
            report = f"""
CARDIOAI REPORT
Risk Score: {st.session_state['prob']:.2f}
Recommendation: Follow medical checkup.
"""
            st.download_button("Download", report, "report.txt")

# ================= QA =================
elif menu == "Hỏi đáp":

    st.subheader("Hỏi đáp")

    q = st.text_input("Nhập câu hỏi")

    if q:
        q = q.lower()
        if "tim" in q:
            st.write("Nguy cơ bệnh tim liên quan huyết áp, cholesterol.")
        elif "đột quỵ" in q:
            st.write("Đột quỵ liên quan huyết áp và đường huyết.")
        else:
            st.write("Chỉ hỗ trợ bệnh tim và đột quỵ.")
