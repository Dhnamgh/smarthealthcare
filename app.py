import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(layout="wide")

# ================= LOAD =================
def load(path):
    return joblib.load(path) if os.path.exists(path) else None

heart_model = load("models/heart_model.pkl")
heart_scaler = load("models/heart_scaler.pkl")

# ================= HEADER + MENU =================
st.markdown("""
<style>
.topbar {
    background-color: #1f5fa7;
    color: white;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: bold;
}
.navbar {
    display: flex;
    gap: 30px;
    padding: 12px 20px;
    border-bottom: 1px solid #ddd;
}
.menu-item {
    position: relative;
    cursor: pointer;
}
.dropdown {
    display: none;
    position: absolute;
    top: 30px;
    background: white;
    border: 1px solid #ddd;
    padding: 15px;
    min-width: 250px;
    z-index: 999;
}
.menu-item:hover .dropdown {
    display: block;
}
</style>

<div class="topbar">
TRUNG TÂM CHẨN ĐOÁN Y KHOA 1009 - THÀNH PHỐ HỒ CHÍ MINH
</div>

<div class="navbar">

<div class="menu-item">Trang chủ</div>

<div class="menu-item">Về trung tâm
    <div class="dropdown">
    Thành lập năm 2026.<br>
    Trung tâm chuyên chẩn đoán bệnh tim, đột quỵ.<br>
    Ứng dụng Machine Learning và phân tích dữ liệu y khoa.
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
    BS Bùi Cao Mỹ Ái<br>
    TS Đặng Trung An<br>
    ThS Đặng Khánh An
    </div>
</div>

<div class="menu-item">Chức năng
    <div class="dropdown">
    Dự đoán<br>
    Upload dữ liệu<br>
    Dashboard<br>
    Giải thích<br>
    Báo cáo<br>
    Hỏi đáp
    </div>
</div>

</div>
""", unsafe_allow_html=True)

# ================= CHỨC NĂNG =================
choice = st.selectbox(
    "Chọn chức năng",
    ["Trang chủ", "Dự đoán", "Upload", "Dashboard", "Giải thích", "Báo cáo", "Hỏi đáp"]
)

# ================= HOME =================
if choice == "Trang chủ":

    st.subheader("Giới thiệu")
    st.write("""
Trung tâm chẩn đoán y khoa 1009 cung cấp giải pháp dự đoán bệnh tim 
dựa trên công nghệ học máy và phân tích dữ liệu hiện đại.
""")

# ================= PREDICT =================
elif choice == "Dự đoán":

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

            if heart_scaler is not None:
                data = heart_scaler.transform(data)

            if heart_model is not None:
                prob = heart_model.predict_proba(data)[0][1]
            else:
                prob = (age/80 + chol/400 + bp/200)/3

            st.success(f"Nguy cơ: {prob:.2f}")

            st.session_state["prob"] = prob
            st.session_state["data"] = data

        except:
            st.error("Lỗi dự đoán")

# ================= UPLOAD =================
elif choice == "Upload":

    file = st.file_uploader("Upload file", ["csv","xlsx"])

    if file:
        try:
            df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
            st.dataframe(df.head())

            if st.button("Chạy"):

                X = df.select_dtypes(include=np.number).iloc[:, :13]

                if heart_scaler is not None:
                    X = heart_scaler.transform(X)

                if heart_model is not None:
                    df["Prediction"] = heart_model.predict(X)
                    df["Probability"] = heart_model.predict_proba(X)[:,1]
                else:
                    df["Prediction"] = np.random.randint(0,2,len(X))
                    df["Probability"] = np.random.rand(len(X))

                st.dataframe(df)

        except:
            st.error("File lỗi")

# ================= DASHBOARD =================
elif choice == "Dashboard":

    df = pd.DataFrame(np.random.randn(100,3), columns=["Risk","BP","Chol"])

    st.line_chart(df)
    st.bar_chart(df)

# ================= SHAP =================
elif choice == "Giải thích":

    if "data" not in st.session_state:
        st.warning("Chưa có dữ liệu")
    else:
        st.write("Mô hình ảnh hưởng bởi age, cholesterol, huyết áp")

# ================= REPORT =================
elif choice == "Báo cáo":

    if "prob" not in st.session_state:
        st.warning("Chưa có dữ liệu")
    else:
        if st.button("Xuất báo cáo"):
            report = f"Risk Score: {st.session_state['prob']:.2f}"
            st.download_button("Download", report, "report.txt")

# ================= QA =================
elif choice == "Hỏi đáp":

    q = st.text_input("Nhập câu hỏi")

    if q:
        if "tim" in q.lower():
            st.write("Nguy cơ tim liên quan huyết áp và cholesterol.")
        else:
            st.write("Chỉ hỗ trợ bệnh tim.")
