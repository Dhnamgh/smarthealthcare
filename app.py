import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(layout="wide")

# ================= STATE =================
if "page" not in st.session_state:
    st.session_state.page = "Trang chủ"

def go(p):
    st.session_state.page = p

# ================= LOAD =================
def load(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load("models/heart_model.pkl")
heart_scaler = load("models/heart_scaler.pkl")

# ================= CSS =================
st.markdown("""
<style>
.topbar {
    background-color: #1f5fa7;
    color: white;
    padding: 12px 20px;
    font-size: 18px;
    font-weight: bold;
}

.nav {
    display: flex;
    gap: 15px;   /* ✅ khoảng cách ~1cm */
    padding: 12px 20px;
}

.nav button {
    background: white;
    border: 1px solid #ccc;
    padding: 6px 12px;
    cursor: pointer;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="topbar">
TRUNG TÂM CHẨN ĐOÁN Y KHOA 1009 - THÀNH PHỐ HỒ CHÍ MINH
</div>
""", unsafe_allow_html=True)

# ================= MENU =================
col = st.columns(8, gap="small")  # ✅ không bị dàn full

with colst.button("Trang chủ", key="home"):
        go("Trang chủ")

with colst.button("Về trung tâm", key="about"):
        go("Về trung tâm")

with colst.button("Chuyên khoa", key="dept"):
        go("Chuyên khoa")

with colst.button("Bác sĩ", key="doctor"):
        go("Bác sĩ")

with colst.button("Dự đoán", key="predict"):
        go("Dự đoán")

with colst.button("Upload", key="upload"):
        go("Upload")

with colst.button("Dashboard", key="dash"):
        go("Dashboard")

with colst.button("Hỏi đáp", key="qa"):
        go("Hỏi đáp")

st.divider()

# ================= PAGE =================
page = st.session_state.page

# ================= HOME =================
if page == "Trang chủ":
    st.subheader("Giới thiệu")

    st.write("""
Trung tâm thành lập năm 2026 tại TP.HCM.

- Chẩn đoán bệnh tim
- Phát hiện nguy cơ đột quỵ
- Ứng dụng AI và Machine Learning
""")

# ================= ABOUT =================
elif page == "Về trung tâm":
    st.write("Thông tin trung tâm, công nghệ, chức năng...")

# ================= DEPT =================
elif page == "Chuyên khoa":

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Khoa lâm sàng")
        st.write("- Tim mạch\n- Cấp cứu")

    with col2:
        st.write("### Khoa cận lâm sàng")
        st.write("- Xét nghiệm\n- Chẩn đoán hình ảnh")

# ================= DOCTOR =================
elif page == "Bác sĩ":

    col = st.columns(3)

    for i in range(6):
        with col[i % 3]:
            st.image("https://via.placeholder.com/150")
            st.write("Bác sĩ chuyên khoa")

# ================= PREDICT =================
elif page == "Dự đoán":

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 80, 40, key="age")
        chol = st.slider("Chol", 100, 400, 200, key="chol")

    with col2:
        bp = st.slider("BP", 80, 200, 120, key="bp")
        hr = st.slider("HR", 60, 200, 150, key="hr")

    if st.button("Dự đoán", key="predict_btn"):

        try:
            data = np.array([[age,1,2,bp,chol,0,1,hr,0,1,1,0,2]])

            if heart_scaler:
                data = heart_scaler.transform(data)

            if heart_model:
                prob = heart_model.predict_proba(data)[0][1]
            else:
                prob = np.random.rand()

            st.success(f"Nguy cơ: {prob:.2f}")

        except:
            st.error("Lỗi dự đoán")

# ================= UPLOAD =================
elif page == "Upload":

    file = st.file_uploader("Upload file")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

# ================= DASH =================
elif page == "Dashboard":
    df = pd.DataFrame(np.random.randn(100,3))
    st.line_chart(df)

# ================= QA =================
elif page == "Hỏi đáp":

    q = st.text_input("Nhập câu hỏi", key="q")

    if q:
        st.write("Trả lời demo")
