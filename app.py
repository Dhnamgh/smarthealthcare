import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(layout="wide")

# ================= STATE =================
if "page" not in st.session_state:
    st.session_state.page = "Trang chủ"

def go(page):
    st.session_state.page = page

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
    font-size: 20px;
    font-weight: bold;
}

.navbar {
    display: flex;
    gap: 30px;
    padding: 12px 20px;
    border-bottom: 1px solid #ddd;
    align-items: center;
}

.menu-item {
    position: relative;
    cursor: pointer;
    font-weight: 500;
}

.dropdown {
    display: none;
    position: absolute;
    top: 25px;
    left: 0;
    background: white;
    border: 1px solid #ddd;
    padding: 10px;
    min-width: 250px;
    z-index: 999;
}

.menu-item:hover .dropdown {
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="topbar">
TRUNG TÂM CHẨN ĐOÁN Y KHOA 1009 - THÀNH PHỐ HỒ CHÍ MINH
</div>
""", unsafe_allow_html=True)

# ================= NAVIGATION =================
col = st.columns(8)

with col[0]:
    if st.button("Trang chủ"):
        go("Trang chủ")

with col[1]:
    if st.button("Về trung tâm"):
        go("Về trung tâm")

with col[2]:
    if st.button("Chuyên khoa"):
        go("Chuyên khoa")

with col[3]:
    if st.button("Bác sĩ"):
        go("Bác sĩ")

with col[4]:
    if st.button("Dự đoán"):
        go("Dự đoán")

with col[5]:
    if st.button("Upload"):
        go("Upload")

with col[6]:
    if st.button("Dashboard"):
        go("Dashboard")

with col[7]:
    if st.button("Hỏi đáp"):
        go("Hỏi đáp")

st.divider()

# ================= PAGE =================
page = st.session_state.page

# ================= HOME =================
if page == "Trang chủ":
    st.subheader("Giới thiệu")

    st.write("""
Trung tâm chẩn đoán y khoa 1009 được thành lập năm 2026.

Chức năng:
- Chẩn đoán bệnh tim mạch
- Phát hiện sớm nguy cơ đột quỵ

Công nghệ:
- Machine Learning
- Phân tích dữ liệu y tế
""")

# ================= ABOUT =================
elif page == "Về trung tâm":
    st.subheader("Về trung tâm")

    st.write("""
Trung tâm chẩn đoán y khoa 1009 được thành lập năm 2026 tại TP.HCM.

Ứng dụng công nghệ:
- AI y tế
- Machine Learning
- Dashboard phân tích

Mục tiêu:
- Hỗ trợ chẩn đoán sớm
- Nâng cao chất lượng điều trị
""")

# ================= DEPARTMENT =================
elif page == "Chuyên khoa":

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Khoa lâm sàng")
        st.write("""
- Tim mạch  
- Cấp cứu  
- Chấn thương chỉnh hình  
""")

        st.write("### Khoa hỗ trợ")
        st.write("""
- Gây mê hồi sức  
- Hậu môn trực tràng  
""")

    with col2:
        st.write("### Khoa cận lâm sàng")
        st.write("""
- Chẩn đoán hình ảnh  
- Xét nghiệm  
- Nội soi  
- Vi sinh  
""")

# ================= DOCTOR =================
elif page == "Bác sĩ":

    col = st.columns(3)

    for i in range(6):
        with col[i % 3]:
            st.image("https://via.placeholder.com/150")
            st.write("BS Chuyên khoa")
            st.button("Xem hồ sơ", key=f"btn{i}")

# ================= PREDICT =================
elif page == "Dự đoán":

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 80, 40)
        chol = st.slider("Cholesterol", 100, 400, 200)

    with col2:
        bp = st.slider("Blood Pressure", 80, 200, 120)
        hr = st.slider("Heart Rate", 60, 200, 150)

    if st.button("Dự đoán"):

        try:
            data = np.array([[age,1,2,bp,chol,0,1,hr,0,1.0,1,0,2]])

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

# ================= DASHBOARD =================
elif page == "Dashboard":

    df = pd.DataFrame(np.random.randn(100,3), columns=["A","B","C"])
    st.line_chart(df)

# ================= QA =================
elif page == "Hỏi đáp":

    q = st.text_input("Nhập câu hỏi")

    if q:
        if "tim" in q.lower():
            st.write("Nguy cơ bệnh tim liên quan huyết áp.")
        else:
            st.write("Chỉ hỗ trợ bệnh tim")
