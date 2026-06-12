import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ================= LOAD MODEL =================
def load(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load("models/heart_model.pkl")
heart_scaler = load("models/heart_scaler.pkl")

# ================= HEADER + NAVBAR =================
st.markdown("""
<style>
.header-top {
    background-color: #1f5fa7;
    color: white;
    padding: 8px 20px;
    font-size: 14px;
}
.header-main {
    display: flex;
    align-items: center;
    padding: 15px 20px;
    border-bottom: 1px solid #ddd;
}
.menu-bar {
    display: flex;
    gap: 30px;
    padding: 10px 20px;
    border-bottom: 1px solid #ddd;
}
.menu-item {
    position: relative;
    cursor: pointer;
    font-weight: 500;
}
.dropdown {
    display: none;
    position: absolute;
    top: 30px;
    left: 0;
    background: white;
    padding: 15px;
    border: 1px solid #ddd;
    min-width: 300px;
    z-index: 999;
}
.menu-item:hover .dropdown {
    display: block;
}
.card {
    background: #f5f7fa;
    padding: 20px;
    border-radius: 10px;
}
</style>

<div class="header-top">
Trung tâm phục vụ cộng đồng | Tư vấn - Dịch vụ - Công nghệ y tế
</div>

<div class="header-main">
<h3>TRUNG TÂM CHẨN ĐOÁN Y KHOA 1009 - THÀNH PHỐ HỒ CHÍ MINH</h3>
</div>

<div class="menu-bar">

<div class="menu-item">Trang chủ</div>

<div class="menu-item">Về trung tâm
<div class="dropdown">
<p>Thành lập năm 2026.</p>
<p>Chuyên chẩn đoán bệnh lý tim mạch, đột quỵ.</p>
<p>Ứng dụng Machine Learning và phân tích dữ liệu y khoa.</p>
</div>
</div>

<div class="menu-item">Chuyên khoa
<div class="dropdown">
<b>Khoa lâm sàng</b><br>
Tim mạch<br>
Cấp cứu<br>
Chấn thương chỉnh hình<br><br>
<b>Khoa cận lâm sàng</b><br>
Chẩn đoán hình ảnh<br>
Xét nghiệm<br>
Vi sinh
</div>
</div>

<div class="menu-item">Bác sĩ
<div class="dropdown">
Danh sách bác sĩ chuyên khoa<br>
Theo lĩnh vực chẩn đoán<br>
Theo kinh nghiệm
</div>
</div>

<div class="menu-item">Dịch vụ</div>
<div class="menu-item">Thư viện sức khỏe</div>
<div class="menu-item">Tin tức & Sự kiện</div>

</div>
""", unsafe_allow_html=True)

# ================= MAIN MENU =================
menu = st.sidebar.radio("", [
    "Trang chủ",
    "Dự đoán",
    "Upload dữ liệu",
    "Dashboard",
    "Giải thích",
    "Báo cáo",
    "Hỏi đáp"
])

# ================= HOME =================
if menu == "Trang chủ":

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Giới thiệu")
        st.write("""
Trung tâm chẩn đoán y khoa 1009 cung cấp giải pháp dự đoán bệnh tim
dựa trên công nghệ học máy và phân tích dữ liệu hiện đại.
""")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1588776814546-ec7e5ef88b14")

# ================= PREDICT =================
elif menu == "Dự đoán":

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
            if heart_scaler:
                data = heart_scaler.transform(data)

            if heart_model:
                prob = heart_model.predict_proba(data)[0][1]
            else:
                prob = (age/80 + chol/400 + bp/200) / 3

            st.success(f"Nguy cơ: {prob:.2f}")

            st.session_state["prob"] = prob
            st.session_state["data"] = data

        except:
            st.error("Lỗi dự đoán")

# ================= UPLOAD =================
elif menu == "Upload dữ liệu":

    file = st.file_uploader("Upload file", ["csv","xlsx"])

    if file:
        try:
            df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
            df.columns = df.columns.str.lower()
            st.dataframe(df.head())

            if st.button("Chạy"):

                X = df.select_dtypes(include=np.number).iloc[:, :13]

                if heart_scaler:
                    X = heart_scaler.transform(X)

                if heart_model:
                    df["Prediction"] = heart_model.predict(X)
                    df["Probability"] = heart_model.predict_proba(X)[:,1]
                else:
                    df["Prediction"] = np.random.randint(0,2,len(X))
                    df["Probability"] = np.random.rand(len(X))

                st.dataframe(df)

        except:
            st.error("File lỗi")

# ================= DASHBOARD =================
elif menu == "Dashboard":

    data = pd.DataFrame(np.random.randn(100,3), columns=["Risk","Cholesterol","BP"])

    col1, col2 = st.columns(2)

    with col1:
        st.line_chart(data)

    with col2:
        st.bar_chart(data)

# ================= SHAP =================
elif menu == "Giải thích":

    if "data" not in st.session_state:
        st.warning("Chưa có dữ liệu")
    else:
        try:
            import shap
            explainer = shap.TreeExplainer(heart_model)
            sv = explainer.shap_values(st.session_state["data"])

            fig = plt.figure()
            shap.summary_plot(sv[1], st.session_state["data"], show=False)
            st.pyplot(fig)
        except:
            st.warning("Không hỗ trợ")

# ================= REPORT =================
elif menu == "Báo cáo":

    if "prob" not in st.session_state:
        st.warning("Chưa có dữ liệu")
    else:
        if st.button("Tạo báo cáo"):
            report = f"Risk Score: {st.session_state['prob']:.2f}"
            st.download_button("Download", report, "report.txt")

# ================= QA =================
elif menu == "Hỏi đáp":

    q = st.text_input("Hỏi gì đó")

    if q:
        if "tim" in q.lower():
            st.write("Bệnh tim liên quan huyết áp và cholesterol.")
        else:
            st.write("Chỉ hỗ trợ bệnh tim.")
